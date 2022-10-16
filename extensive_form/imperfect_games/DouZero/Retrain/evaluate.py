import os 
import argparse

from douzero.evaluation.simulation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Evaluation')
    parser.add_argument('--landlord', type=str,
            default='/data/zelei/DouZero_lasso_0.06/baselines/douzero_WP/landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='/data/zelei/DouZero_lasso_0.06/baselines/douzero_WP/landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='/data/zelei/DouZero_lasso_0.06/baselines/douzero_WP/landlord_down.ckpt')
    parser.add_argument('--retrain_landlord', type=str,
            default='/data/zelei/DouZero_lasso_0.06/DouZero-main/retrain_checkpoints/douzero/landlord_weights_83200.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.eval_data,
             args.num_workers)
    
    evaluate(args.retrain_landlord,
             args.landlord_up,
             args.landlord_down,
             args.eval_data,
             args.num_workers)
