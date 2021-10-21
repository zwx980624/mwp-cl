# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
import time
import torch
import torch.optim
from src.expressions_transfer import *
from transformers import AdamW, get_linear_schedule_with_warmup

import tqdm
import json
import logging
import argparse
import shutil
import random
import numpy as np

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def alpha_schedule(beg, end, epoch, final_val):
    if epoch <= beg:
        return 0
    elif epoch >= end:
        return  final_val
    else:
        return float(final_val) * (epoch - beg) / (end - beg)

def make_pair(data, is_eval=False):
    items = data["pairs"]
    generate_nums = data["generate_nums"]
    copy_nums = data["copy_nums"]

    temp_pairs = []
    for p in items:
        if not is_eval:
            temp_pairs.append((p["tokens"], from_infix_to_prefix(p["expression"])[:MAX_OUTPUT_LENGTH], p["nums"], p["num_pos"]))
        else:
            temp_pairs.append((p["tokens"], from_infix_to_prefix(p["expression"]), p["nums"], p["num_pos"]))
    pairs = temp_pairs
    return pairs, generate_nums, copy_nums

def initial_model(output_lang, embedding_size, hidden_size, args, copy_nums, generate_nums):
    encoder = EncoderBert(hidden_size=hidden_size, auto_transformer=False, 
                        bert_pretrain_path=args.bert_pretrain_path, dropout=args.dropout)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        input_size=len(generate_nums), dropout=args.dropout)
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size, dropout=args.dropout)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size, dropout=args.dropout)

    if args.model_reload_path != '' and os.path.exists(args.model_reload_path):
        encoder.load_state_dict(torch.load(os.path.join(args.model_reload_path, "encoder.ckpt")))
        pred = torch.load(os.path.join(args.model_reload_path, "predict.ckpt"))
        gene = torch.load(os.path.join(args.model_reload_path, "generate.ckpt"))
        if args.finetune_from_trainset != "": # alignment finetune output vocab
            logger.info("alignment finetune output vocab with {}".format(args.finetune_from_trainset))
            from_train_data = json.load(open(os.path.join(args.data_dir, args.finetune_from_trainset), 'r', encoding='utf-8'))
            from_pairs_trained, from_generate_nums, from_copy_nums = make_pair(from_train_data)
            use_bert = True
            _, from_output_lang, _, _, _ = prepare_data(from_pairs_trained, (), 5, from_generate_nums, from_copy_nums, tree=True, use_bert=use_bert, auto_transformer=False, bert_pretrain_path=args.bert_pretrain_path)
            op_weight = None
            op_bias = None
            gene_embed_weight = None
            for i in range(output_lang.num_start): # op
                op = output_lang.index2word[i]
                from_idx = from_output_lang.word2index[op]
                if op_weight == None:
                    op_weight = pred["ops.weight"][from_idx:from_idx+1, :]
                    op_bias = pred["ops.bias"][from_idx:from_idx+1]
                    gene_embed_weight = gene["embeddings.weight"][from_idx:from_idx+1, :]
                else:
                    op_weight = torch.cat([op_weight, pred["ops.weight"][from_idx:from_idx+1, :]], dim=0)
                    op_bias = torch.cat([op_bias, pred["ops.bias"][from_idx:from_idx+1]], dim=0)
                    gene_embed_weight = torch.cat([gene_embed_weight, gene["embeddings.weight"][from_idx:from_idx+1, :]], dim=0)
            pred["ops.weight"] = op_weight
            pred["ops.bias"] = op_bias
            gene["embeddings.weight"] = gene_embed_weight
            embedding_weight = None
            for i in generate_num_ids: # constant
                constant = output_lang.index2word[i]
                const_emb = None
                if constant not in from_output_lang.word2index:
                    const_emb = nn.Parameter(torch.randn(1, 1, hidden_size))
                    if USE_CUDA:
                        const_emb = const_emb.cuda()
                else:
                    from_idx = from_output_lang.word2index[constant] - from_output_lang.num_start
                    const_emb = pred["embedding_weight"][:,from_idx:from_idx+1,:]
                if embedding_weight == None:
                    embedding_weight = const_emb
                else:
                    embedding_weight = torch.cat([embedding_weight, const_emb], dim=1)
            pred["embedding_weight"] = embedding_weight
        predict.load_state_dict(pred)
        generate.load_state_dict(gene)
        merge.load_state_dict(torch.load(os.path.join(args.model_reload_path, "merge.ckpt")))

    return encoder, predict, generate, merge

def train_model(args, train_pairs, test_pairs, generate_num_ids, 
                    encoder, predict, generate, merge, output_lang):
    batch_size = args.batch_size
    need_optimized_parameters = []
    for module in [encoder, predict, generate, merge]:
        need_optimized_parameters += [p for n, p in module.named_parameters() if p.requires_grad]

    contra_pair = None
    subtree_pos_pair = None
    
    logger.info("Loading contra pair file: {}".format(args.contra_pair))
    contra_pair = json.load(open(os.path.join(args.data_dir, args.contra_pair), 'r', encoding='utf-8'))
    if isinstance(contra_pair, dict):
        subtree_pos_pair = contra_pair["pos"]
        contra_pair = contra_pair["pairs"]

    t_total = (len(contra_pair) // batch_size + 1) * args.n_epochs
    logger.info("Num of Training Data = {}".format(len(contra_pair)))
    logger.info("Total Steps = {}".format(t_total))
    optimizer = AdamW([{'params': need_optimized_parameters, 'weight_decay': 0.0}], lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    best_metric = (0, 0, 0)
    best_value_acc_ls = [0.0 for _ in range(args.n_save_ckpt)]
    best_equ_acc_ls = [0.0 for _ in range(args.n_save_ckpt)]
    best_metric_ls = [(0, 0, 0) for _ in range(args.n_save_ckpt)]
    best_epoch_ls = [-1 for _ in range(args.n_save_ckpt)]

    with open(os.path.join(args.output_dir, 'training_args.txt'), 'w', encoding='utf-8') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))

    logger.info("Training start...")

    run_steps = 0
    for epoch in range(args.n_epochs):        
        loss_total = 0
        contra_loss_total = 0
        expr_loss_total = 0

        if args.contra_common_tree_pair:
            input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, subtree_pos_pair_batches = prepare_contra_train_batch(train_pairs, batch_size, contra_pair, subtree_pos_pair, args, args.neg_sample, args.neg_sample_from_pair_file) # input_batches = [(input_batch1, input_batch2, neg_batch, ...), ...] ...
        else:
            input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_contra_train_batch(train_pairs, batch_size, contra_pair, subtree_pos_pair, args, args.neg_sample, args.neg_sample_from_pair_file) # input_batches = [(input_batch1, input_batch2, neg_batch, ...), ...] ...

        logger.info("epoch: {}".format(epoch + 1))
        start = time.time()

        for idx in range(len(input_lengths)):
            # alpha warmup
            if args.alpha_warmup:
                alpha = alpha_schedule(args.warmup_begin, args.warmup_end, run_steps, args.alpha)
            else:
                alpha = args.alpha
            
            if args.contra_common_tree_pair:
                loss, contra_loss, expr_loss = contra_train_tree(
                    input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                    num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
                    output_lang, num_pos_batches[idx], args, alpha, args.contra_loss_func, subtree_pos_pair_batches[idx])
            else:
                loss, contra_loss, expr_loss = contra_train_tree(
                    input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                    num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
                    output_lang, num_pos_batches[idx], args, alpha, args.contra_loss_func)
            
            torch.nn.utils.clip_grad_norm_(need_optimized_parameters, args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            encoder.zero_grad()
            predict.zero_grad()
            generate.zero_grad()
            merge.zero_grad()

            contra_loss_total += contra_loss
            expr_loss_total += expr_loss

            loss_total += loss
            run_steps += 1

            if run_steps % args.logging_steps == 0:
                logger.info("step: {}, lr: {}, loss: {}, c_alpha: {}, c_loss: {}, e_loss: {}".format(run_steps, scheduler.get_last_lr()[0], loss_total/(idx+1), alpha, contra_loss_total/(idx+1), expr_loss_total/(idx+1)))


        logger.info("loss: {}, contra_loss: {}, expr_loss: {}".format(loss_total / len(input_lengths), contra_loss_total / len(input_lengths), expr_loss_total / len(input_lengths)))
        logger.info("training time: {}".format(time_since(time.time() - start)))
        logger.info("--------------------------------")
        del input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches

        if epoch % args.n_val == 0 or epoch > args.n_epochs - 5:
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            start = time.time()

            for nd in range(len(test_pairs)):
                value_ac_0 = 0
                equation_ac_0 = 0
                eval_total_0 = 0
                for test_batch in test_pairs[nd]:
                    test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                            merge, output_lang, test_batch[5], beam_size=beam_size)
                    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
                    del test_res
                    if val_ac:
                        value_ac_0 += 1
                        value_ac += 1
                    if equ_ac:
                        equation_ac_0 += 1
                        equation_ac += 1
                    eval_total_0 += 1
                    eval_total += 1
                logger.info("{}, {}, {}".format(equation_ac_0, value_ac_0, eval_total_0))
                logger.info("test_answer_acc: {}, {}".format(float(equation_ac_0) / eval_total, float(value_ac_0) / eval_total_0))

            logger.info("{}, {}, {}".format(equation_ac, value_ac, eval_total))
            logger.info("test_answer_acc: {}, {}".format(float(equation_ac) / eval_total, float(value_ac) / eval_total))
            logger.info("best_answer_acc: {}, {}".format(max(best_equ_acc_ls), max(best_value_acc_ls)))
            logger.info("testing time: {}".format(time_since(time.time() - start)))
            logger.info("------------------------------------------------------")
            if float(value_ac) / eval_total > min(best_value_acc_ls):
                if float(value_ac) / eval_total > max(best_value_acc_ls):
                    best_metric = (equation_ac, value_ac, eval_total)
                min_pos = best_value_acc_ls.index(min(best_value_acc_ls))
                best_value_acc_ls[min_pos] = float(value_ac) / eval_total
                best_equ_acc_ls[min_pos] = float(equation_ac) / eval_total
                best_metric_ls[min_pos] = (equation_ac, value_ac, eval_total)
                logger.info("delete checkpoint: epoch {}".format(best_epoch_ls[min_pos]))
                if best_epoch_ls[min_pos] != -1:
                    shutil.rmtree(os.path.join(args.output_dir, "epoch_{}".format(best_epoch_ls[min_pos])))
                logger.info("saving best checkpoint")
                best_epoch_ls[min_pos] = epoch
                if os.path.exists(os.path.join(args.output_dir, "epoch_{}".format(epoch))):
                     shutil.rmtree(os.path.join(args.output_dir, "epoch_{}".format(epoch)))
                os.makedirs(os.path.join(args.output_dir, "epoch_{}".format(epoch)))
                torch.save(encoder.state_dict(), os.path.join(args.output_dir, "epoch_{}".format(epoch), "encoder.ckpt"))
                torch.save(predict.state_dict(), os.path.join(args.output_dir, "epoch_{}".format(epoch), "predict.ckpt"))
                torch.save(generate.state_dict(), os.path.join(args.output_dir, "epoch_{}".format(epoch), "generate.ckpt"))
                torch.save(merge.state_dict(), os.path.join(args.output_dir, "epoch_{}".format(epoch), "merge.ckpt"))
    return best_metric

def test_model(args, test_pairs, generate_num_ids, encoder, predict, generate, merge, output_lang, beam_size):
    epochs = os.listdir(os.path.join(args.output_dir))
    for epoch in epochs:
        if not epoch.startswith('epoch'):
            continue
        logger.info("testing -> " + os.path.join(args.output_dir, epoch))
        encoder.load_state_dict(torch.load(os.path.join(args.output_dir, epoch, "encoder.ckpt")))
        predict.load_state_dict(torch.load(os.path.join(args.output_dir, epoch, "predict.ckpt")))
        generate.load_state_dict(torch.load(os.path.join(args.output_dir, epoch, "generate.ckpt")))
        merge.load_state_dict(torch.load(os.path.join(args.output_dir, epoch, "merge.ckpt")))
        for test_pair in test_pairs:
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            start = time.time()
            for test_batch in test_pair:
                test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                        merge, output_lang, test_batch[5], beam_size=beam_size)
                val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
                del test_res
                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1
                eval_total += 1
            logger.info("{}, {}, {}".format(equation_ac, value_ac, eval_total))
            logger.info("test_answer_acc: {}, {}".format(float(equation_ac) / eval_total, float(value_ac) / eval_total))
            logger.info("testing time: {}".format(time_since(time.time() - start)))
            logger.info("------------------------------------------------------")

    return (equation_ac, value_ac, eval_total)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output_dir', default='', type=str, required=True, help='Model Saved Path, Output Directory')
    parser.add_argument('--bert_pretrain_path', default='', type=str, required=True)
    parser.add_argument('--train_file', default='', type=str, required=True)
    parser.add_argument('--contra_pair', default='', type=str, required=True, help='Contrastive Pair File')
    

    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--dev_file_1', default='Math_23K_mbert_token_val.json', type=str)
    parser.add_argument('--test_file_1', default='Math_23K_mbert_token_test.json', type=str)
    parser.add_argument('--dev_file_2', default='MathQA_mbert_token_val.json', type=str)
    parser.add_argument('--test_file_2', default='MathQA_mbert_token_test.json', type=str)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--max_grad_norm', default=3.0, type=int)

    parser.add_argument('--n_save_ckpt', default=5, type=int, help='totally save $n_save_ckpt best ckpts')
    parser.add_argument('--n_val', default=5, type=int, help='conduct validation every $n_val epochs')
    parser.add_argument('--logging_steps', default=100, type=int)

    parser.add_argument('--embedding_size', default=128, type=int, help='Embedding size')
    parser.add_argument('--hidden_size', default=512, type=int, help='Hidden size')
    parser.add_argument('--beam_size', default=5, type=int, help='Beam size')

    parser.add_argument('--contra_loss_func', default='margin', type=str, help='margin, ...')
    parser.add_argument('--contra_loss_margin', default=0.2, type=float)
    parser.add_argument('--contra_common_tree_pair', action='store_true', help='contra learn use common_tree pair')

    parser.add_argument('--neg_sample', default=1 type=int, help="Neg sample num for contra learn")
    parser.add_argument('--neg_sample_from_pair_file', action='store_true', help='if true: neg samples from pair file, else: random neg sample')
    parser.add_argument('--neg_no_expr_loss', action='store_true', help='neg samples not compute expression loss')
    
    parser.add_argument('--alpha', default=0.2, type=float, help="contra_loss weight")
    parser.add_argument('--alpha_warmup', action='store_true', help='alpha warmup')
    parser.add_argument('--warmup_begin', type=int, help="alpha=0 until epoch warmup_begin")
    parser.add_argument('--warmup_end', type=int, help="alpha=alpha until epoch warmup_end")

    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--seed', default=42, type=int, help='universal seed')

    parser.add_argument('--only_test', action='store_true')

    parser.add_argument('--model_reload_path', default='', type=str)
    parser.add_argument('--finetune_from_trainset', default='', type=str, help='train_file which pretrained model used, important for alignment output vocab')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args)

    if os.path.exists(os.path.join(args.output_dir, "log.txt")) and not args.only_test:
        # print("remove log file")
        os.remove(os.path.join(args.output_dir, "log.txt"))
    if args.only_test:
        handler = logging.FileHandler(os.path.join(args.output_dir, "log_test.txt"))
    else:
        handler = logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    embedding_size  =   args.embedding_size
    hidden_size     =   args.hidden_size   
    beam_size       =   args.beam_size     

    train_data = json.load(open(os.path.join(args.data_dir, args.train_file), 'r', encoding='utf-8'))
    val_data1 = json.load(open(os.path.join(args.data_dir, args.dev_file_1), 'r', encoding='utf-8'))
    test_data1 = json.load(open(os.path.join(args.data_dir, args.test_file_1), 'r', encoding='utf-8'))
    val_data2 = json.load(open(os.path.join(args.data_dir, args.dev_file_2), 'r', encoding='utf-8'))
    test_data2 = json.load(open(os.path.join(args.data_dir, args.test_file_2), 'r', encoding='utf-8'))

    pairs_trained, generate_nums, copy_nums = make_pair(train_data, False)
    pairs_tested1, _, _ = make_pair(test_data1, True)
    pairs_valed1, _, _ = make_pair(val_data1, True)
    pairs_tested2, _, _ = make_pair(test_data2, True)
    pairs_valed2, _, _ = make_pair(val_data2, True)

    use_bert = True
    input_lang, output_lang, train_pairs, (test_pairs1, val_pairs1, test_pairs2, val_pairs2), len_bert_token = prepare_data(pairs_trained, (pairs_tested1, pairs_valed1, pairs_tested2, pairs_valed2), 5, generate_nums,
                                                                                copy_nums, tree=True, use_bert=use_bert, auto_transformer=False, bert_pretrain_path=args.bert_pretrain_path)

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])
    encoder, predict, generate, merge = initial_model(output_lang, embedding_size, hidden_size, args, copy_nums, generate_nums)
    if torch.cuda.is_available():
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()

    if not args.only_test:
        train_model(args, train_pairs, (val_pairs1, val_pairs2), generate_num_ids, 
                    encoder, predict, generate, merge, output_lang)
    test_model(args, (test_pairs1, test_pairs2), generate_num_ids,encoder, predict, generate, merge, output_lang, beam_size)

