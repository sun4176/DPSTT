from libs.dataset.data import (
        ROOT, 
        MAX_TRAINING_OBJ, 
        build_dataset, 
        multibatch_collate_fn, 
        convert_one_hot, 
        convert_mask
    )

from libs.utils.logger import AverageMeter
from libs.utils.loss import *
from libs.utils.utility import parse_args, save_checkpoint, adjust_learning_rate
from libs.models.models import STAN

import torch
import numpy as np
import time
import copy
from progress.bar import Bar



def train(trainloader, model, criterion, optimizer, use_cuda, iter_size):

    data_time = AverageMeter()
    loss = AverageMeter()

    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    for batch_idx, data in enumerate(trainloader):

        frames, masks, objs, _ = data
        max_obj = masks.shape[2]-1
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        if use_cuda:
            frames = frames.to(device)
            masks = masks.to(device)

        N, T, C, H, W = frames.size()
        total_loss = 0.0
        for idx in range(N):  # batch loop
            frame = frames[idx]
            mask = masks[idx]
            num_objects = objs[idx]

            keys = []
            vals = []
            for t in range(1, T):
                # memorize
                if t-1 == 0:
                    tmp_mask = mask[t-1:t]
                else:
                    tmp_mask = out                    

                key, val, r4 = model(frame=frame[t-1:t, :, :, :], mask=tmp_mask, num_objects=num_objects)

                keys.append(key)
                vals.append(val)

                # segment
                tmp_key = torch.cat(keys, dim=0)
                tmp_val = torch.cat(vals, dim=0)

                logits, _ = model(frame=frame[t:t+1, :, :, :], keys=tmp_key, values=tmp_val, num_objects=num_objects,
                                    max_obj=max_obj, patch=2)

                out = torch.softmax(logits, dim=1)
                gt = mask[t:t+1]
                
                total_loss = total_loss + criterion(out, gt, num_objects, ref=mask[0:1, :num_objects+1])

            # cycle-consistancy
            key, val, r4 = model(frame=frame[T-1:T, :, :, :], mask=out, num_objects=num_objects)
            keys.append(key)
            vals.append(val)

            cycle_loss = 0.0
            for t in range(T-1, 0, -1): 
                cm = np.transpose(mask[t].detach().cpu().numpy(), [1, 2, 0])
                if convert_one_hot(cm, max_obj).max() == num_objects:
                    tmp_key = keys[t]
                    tmp_val = vals[t]
                    logits, _ = model(frame=frame[0:1, :, :, :], keys=tmp_key, values=tmp_val, num_objects=num_objects,
                                        max_obj=max_obj, patch=2)
                    first_out = torch.softmax(logits, dim=1)
                    cycle_loss += criterion(first_out, mask[0:1], num_objects, ref=mask[t:t+1, :num_objects+1])
            
            total_loss = total_loss + cycle_loss

        total_loss = total_loss / (N * (T-1))

        if isinstance(total_loss, torch.Tensor) and total_loss.item() > 0.0:
            loss.update(total_loss.item(), 1)

            # compute gradient and do SGD step (divided by accumulated steps)
            total_loss /= iter_size
            total_loss.backward()

        if (batch_idx+1) % iter_size == 0:
            optimizer.step()
            model.zero_grad()

        # measure elapsed time
        end = time.time()
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.5f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=loss.val
        )
        bar.next()
    bar.finish()

    return loss.avg





if __name__ == '__main__':

    main()
