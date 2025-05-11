from libs.models.models import STAN
import torch
import numpy as np
import os
import time
from progress.bar import Bar



def test_adaptive_memory(testloader, model, use_cuda, model_name, opt):

    data_time = AverageMeter()
    bar = Bar('Processing adaptive memory testing', max=len(testloader))

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):

            frames, masks, objs, infos = data

            if use_cuda:
                frames = frames.to(device)
                masks = masks.to(device)
                
            frames = frames[0]
            masks = masks[0]
            num_objects = objs[0]
            info = infos[0]
            max_obj = masks.shape[1]-1
            t1 = time.time()

            T, _, H, W = frames.shape
            pred = []
            keys, vals = [], []
            keys_dict, vals_dict = {}, {}
            for t in range(1, T+1):
                if t - 1 == 0:
                    tmp_mask = torch.zeros_like(masks[0:1])
                elif 'frame' in info and t - 1 in info['frame']:
                    # start frame
                    mask_id = info['frame'].index(t - 1)
                    tmp_mask = masks[mask_id:mask_id + 1]
                    num_objects = max(num_objects, tmp_mask.max())
                else:
                    tmp_mask = out

                key, val, r4 = model(frame=frames[t-1:t], mask=tmp_mask, num_objects=num_objects)
                keys.append(key)
                vals.append(val)
                tmp_key = torch.cat(keys, dim=0)
                tmp_val = torch.cat(vals, dim=0)
                keys_dict[t - 1] = key
                vals_dict[t - 1] = val
                frame_idx = t
                logits, _ = model(frame=frames[t-1: t], keys=tmp_key, values=tmp_val, num_objects=num_objects, max_obj=max_obj, 
                                    opt=opt, frame_idx=frame_idx, keys_dict=keys_dict, vals_dict=vals_dict, patch=2, is_testing=True)
                out = torch.softmax(logits, dim=1)
                pred.append(out)

            pred_back = [masks[T-1:T]]
            keys_back, vals_back = [], []
            keys_dict_back, vals_dict_back = {}, {}
            for t in range(T-1, 0, -1):
                if t == T-1:
                    tmp_mask_back = masks[T-1: T]
                else:
                    tmp_mask_back = out_back

                key_back, val_back, r4_back = model(frame=frames[t: t + 1], mask=tmp_mask_back,
                                                    num_objects=num_objects)
                keys_back.append(key_back)
                vals_back.append(val_back)
                tmp_key_back = torch.cat(keys, dim=0)
                tmp_val_back = torch.cat(vals, dim=0)
                keys_dict_back[(T-1)-(t-1)-1] = key_back
                vals_dict_back[(T-1)-(t-1)-1] = val_back             
                frame_idx_back = (T-1)-(t-1)-1
                logits_back, _ = model(frame=frames[t-1: t], keys=tmp_key_back, values=tmp_val_back, num_objects=num_objects, 
                                max_obj=max_obj, opt=opt, frame_idx=frame_idx_back, keys_dict=keys_dict_back, vals_dict=vals_dict_back,
                                patch=2, is_testing=True)
                out_back = torch.softmax(logits_back, dim=1)
                pred_back.append(out_back)

            pred = torch.cat(pred, dim=0)
            pred = pred.detach().cpu().numpy()
            pred_back = torch.cat(pred_back, dim=0)
            pred_back = pred_back.detach().cpu().numpy()
            assert num_objects == 1
            union = []
            for idx in range(len(pred)):
                sum = pred[idx] + pred_back[(len(pred_back)-1-idx)]
                sum[sum > 1] = 1
                union.append(sum)
            union = np.array(union)
            write_mask(union, info, opt, directory=opt.output_dir, model_name='{}_random_adapt'.format(model_name))

            toc = time.time() - t1
            data_time.update(toc, 1)
            bar.suffix  = '({batch}/{size}) Time: {data:.3f}s'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.sum
            )
            bar.next()
        bar.finish()

    return data_time.sum



if __name__ == '__main__':

    models_path = './ckpt/'
    models = [file for file in os.listdir(models_path) if file.endswith('pth.tar')]
    for model_name in models:
        model_path = os.path.join(models_path, model_name)
        model_name = model_name.replace('.', '_')
        main(model_name=model_name, model_path=model_path)
