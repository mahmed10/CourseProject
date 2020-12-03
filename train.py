import argparse

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.parse_config import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
	from apex import amp
except:
	print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
	mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'


# Hyperparameters
hyp = {'giou': 3.54,  # giou loss gain
	'cls': 37.4,  # cls loss gain
	'cls_pw': 1.0,  # cls BCELoss positive_weight
	'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
	'obj_pw': 1.0,  # obj BCELoss positive_weight
	'iou_t': 0.20,  # iou training threshold
	'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
	'lrf': 0.0005,  # final learning rate (with cos scheduler)
	'momentum': 0.937,  # SGD momentum
	'weight_decay': 0.0005,  # optimizer weight decay
	'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
	'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
	'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
	'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
	'degrees': 1.98 * 0,  # image rotation (+/- deg)
	'translate': 0.05 * 0,  # image translation (+/- fraction)
	'scale': 0.05 * 0,  # image scale (+/- gain)
	'shear': 0.641 * 0}  # image shear (+/- deg)

def train(hyp):
	cfg = opt.cfg
	data = opt.data
	epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
	batch_size = opt.batch_size
	accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
	weights = opt.weights  # initial training weights
	imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)

	# Image Sizes
	gs = 32  # (pixels) grid size
	assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
	opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
	if opt.multi_scale:
		if imgsz_min == imgsz_max:
			imgsz_min //= 1.5
			imgsz_max //= 0.667
		grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
		imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
	img_size = imgsz_max  # initialize with max size

	# Configure run
	init_seeds()
	data_dict = parse_data_cfg(data)
	train_path = data_dict['train']
	test_path = data_dict['valid']
	nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
	hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

	# Remove previous results
	for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
		os.remove(f)

	# Initialize model
	model = Darknet(cfg).to(device)

	# Optimizer
	pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
	for k, v in dict(model.named_parameters()).items():
		if '.bias' in k:
			pg2 += [v]  # biases
		elif 'Conv2d.weight' in k:
			pg1 += [v]  # apply weight_decay
		else:
			pg0 += [v]  # all else

	optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
	optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
	optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
	print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
	del pg0, pg1, pg2

	start_epoch = 0
	best_fitness = 0.0

	#do something with weights initialization
	ckpt = torch.load(weights, map_location=device)

	ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
	model.load_state_dict(ckpt['model'], strict=False)

	start_epoch = ckpt['epoch'] + 1
	del ckpt
	

	# Scheduler https://arxiv.org/pdf/1812.01187.pdf
	lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
	scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
	scheduler.last_epoch = start_epoch - 1  # see link below

	# Dataset
	dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
									augment=True,
									hyp=hyp,  # augmentation hyperparameters
									rect=opt.rect,  # rectangular training
									cache_images=opt.cache_images,
									single_cls=opt.single_cls)

	# Dataloader
	batch_size = min(batch_size, len(dataset))
	nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

	dataloader = torch.utils.data.DataLoader(dataset,
									batch_size=batch_size,
									num_workers=nw,
									shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
									pin_memory=True,
									collate_fn=dataset.collate_fn)

	# Testloader
	testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
									hyp=hyp,
									rect=True,
									cache_images=opt.cache_images,
									single_cls=opt.single_cls),
									batch_size=batch_size,
									num_workers=nw,
									pin_memory=True,
									collate_fn=dataset.collate_fn)

	print(nc)
	print(hyp)

	# Model parameters
	model.nc = nc  # attach number of classes to model
	model.hyp = hyp  # attach hyperparameters to model
	model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
	model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights

	# Model EMA
	ema = torch_utils.ModelEMA(model)

	# Start training
	nb = len(dataloader)  # number of batches
	n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
	maps = np.zeros(nc)  # mAP per class
	# torch.autograd.set_detect_anomaly(True)
	results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
	t0 = time.time()
	print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
	print('Using %g dataloader workers' % nw)
	print('Starting training for %g epochs...' % epochs)


	for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
		model.train()

		mloss = torch.zeros(4).to(device)  # mean losses
		print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
		pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar

		for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
			ni = i + nb * epoch  # number integrated batches (since train start)
			imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
			targets = targets.to(device)

			# Burn-in
			if ni <= n_burn:
				xi = [0, n_burn]  # x interp
				model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
				accumulate = max(1, np.interp(ni, xi, [1, 64 / batch_size]).round())
				for j, x in enumerate(optimizer.param_groups):
					# bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
					x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
					x['weight_decay'] = np.interp(ni, xi, [0.0, hyp['weight_decay'] if j == 1 else 0.0])
					if 'momentum' in x:
						x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

			# Forward
			pred = model(imgs)

			# Loss
			loss, loss_items = compute_loss(pred, targets, model)
			if not torch.isfinite(loss):
				print('WARNING: non-finite loss, ending training ', loss_items)
				return results

			# Backward
			loss *= batch_size / 64  # scale loss
			loss.backward()

			# Optimize
			if ni % accumulate == 0:
				optimizer.step()
				optimizer.zero_grad()
				ema.update(model)

			# Print
			mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
			mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
			s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
			pbar.set_description(s)

			# Plot
			if ni < 1:
				f = 'train_batch%g.jpg' % i  # filename
			# end batch ------------------------------------------------------------------------------------------------


		# Update scheduler
		scheduler.step()

		# Process epoch results
		ema.update_attr(model)
		final_epoch = epoch + 1 == epochs
		if not opt.notest or final_epoch:  # Calculate mAP
			is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
			results, maps = test.test(cfg,
									data,
									batch_size=batch_size,
									imgsz=imgsz_test,
									model=ema.ema,
									save_json=final_epoch and is_coco,
									single_cls=opt.single_cls,
									dataloader=testloader,
									multi_label=ni > n_burn)

		# Write
		with open(results_file, 'a') as f:
			f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

		# Tensorboard
		if tb_writer:
			tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
					'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
					'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
			for x, tag in zip(list(mloss[:-1]) + list(results), tags):
				tb_writer.add_scalar(tag, x, epoch)

		# Update best mAP
		fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
		if fi > best_fitness:
			best_fitness = fi

		# Save model
		save = (not opt.nosave) or (final_epoch and not opt.evolve)
		if save:
			with open(results_file, 'r') as f: # create checkpoint
				ckpt = {'epoch': epoch,
						'best_fitness': best_fitness,
						'training_results': f.read(),
						'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
						'optimizer': None if final_epoch else optimizer.state_dict()}

			# Save last, best and delete
			torch.save(ckpt, last)
			if (best_fitness == fi) and not final_epoch:
				torch.save(ckpt, best)
			del ckpt

		# end epoch ----------------------------------------------------------------------------------------------------
	# end training

	n = opt.name
	if len(n):
		n = '_' + n if not n.isnumeric() else n
		fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
		for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
			if os.path.exists(f1):
				os.rename(f1, f2)  # rename
				ispt = f2.endswith('.pt')  # is *.pt
				strip_optimizer(f2) if ispt else None  # strip optimizer
				os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

	print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
	dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
	torch.cuda.empty_cache()
	return results



if __name__ == '__main__':
	#torch.cuda.empty_cache()
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=5)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
	parser.add_argument('--batch-size', type=int, default=5)  # effective bs = batch_size * accumulate = 16 * 4 = 64
	parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny.cfg', help='*.cfg path')
	parser.add_argument('--data', type=str, default='data/dronedata.data', help='*.data path')
	parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
	parser.add_argument('--img-size', nargs='+', type=int, default=[320, 640], help='[min_train, max-train, test]')
	parser.add_argument('--rect', action='store_true', help='rectangular training')
	parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
	parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
	parser.add_argument('--notest', action='store_true', help='only test final epoch')
	parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
	parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
	parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
	parser.add_argument('--weights', type=str, default='weights/yolov3-tiny.pt', help='initial weights path')
	parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
	parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
	parser.add_argument('--adam', action='store_true', help='use adam optimizer')
	parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
	parser.add_argument('--freeze-layers', action='store_true', help='Freeze non-output layers')
	opt = parser.parse_args()
	opt.weights = last if opt.resume and not opt.weights else opt.weights
	#check_git_status()
	opt.cfg = check_file(opt.cfg)  # check file
	opt.data = check_file(opt.data)  # check file
	print(opt)
	opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
	device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
	print(device)

	tb_writer = SummaryWriter(comment=opt.name)
	train(hyp)