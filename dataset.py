from datasets.saliency_db import saliency_db


def get_training_set(opt, spatial_transform, temporal_transform,
					 target_transform):

	assert opt.dataset in ['diem', 'coutrot1', 'coutrot2', 'summe', 'etmd', 'avad']
	print('Creating training dataset: {}'.format(opt.dataset))

	if opt.dataset == 'diem':
		training_data = saliency_db(
			opt.video_path_diem,
			opt.annotation_path_diem_train,
			opt.salmap_path_diem,
			opt.audio_path_diem,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'coutrot1':
		training_data = saliency_db(
			opt.video_path_coutrot1,
			opt.annotation_path_coutrot1_train,
			opt.salmap_path_coutrot1,
			opt.audio_path_coutrot1,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'coutrot2':
		training_data = saliency_db(
			opt.video_path_coutrot2,
			opt.annotation_path_coutrot2_train,
			opt.salmap_path_coutrot2,
			opt.audio_path_coutrot2,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'summe':
		training_data = saliency_db(
			opt.video_path_summe,
			opt.annotation_path_summe_train,
			opt.salmap_path_summe,
			opt.audio_path_summe,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'etmd':
		training_data = saliency_db(
			opt.video_path_etmd,
			opt.annotation_path_etmd_train,
			opt.salmap_path_etmd,
			opt.audio_path_etmd,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'avad':
		training_data = saliency_db(
			opt.video_path_avad,
			opt.annotation_path_avad_train,
			opt.salmap_path_avad,
			opt.audio_path_avad,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)

	return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
					   target_transform):

	assert opt.dataset in ['diem', 'coutrot1', 'coutrot2', 'summe', 'etmd', 'avad']
	print('Creating validation dataset: {}'.format(opt.dataset))

	if opt.dataset == 'diem':
		validation_data = saliency_db(
			opt.video_path_diem,
			opt.annotation_path_diem_test,
			opt.salmap_path_diem,
			opt.audio_path_diem,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'coutrot1':
		validation_data = saliency_db(
			opt.video_path_coutrot1,
			opt.annotation_path_coutrot1_test,
			opt.salmap_path_coutrot1,
			opt.audio_path_coutrot1,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'coutrot2':
		validation_data = saliency_db(
			opt.video_path_coutrot2,
			opt.annotation_path_coutrot2_test,
			opt.salmap_path_coutrot2,
			opt.audio_path_coutrot2,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'summe':
		validation_data = saliency_db(
			opt.video_path_summe,
			opt.annotation_path_summe_test,
			opt.salmap_path_summe,
			opt.audio_path_summe,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'etmd':
		validation_data = saliency_db(
			opt.video_path_etmd,
			opt.annotation_path_etmd_test,
			opt.salmap_path_etmd,
			opt.audio_path_etmd,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'avad':
		validation_data = saliency_db(
			opt.video_path_avad,
			opt.annotation_path_avad_test,
			opt.salmap_path_avad,
			opt.audio_path_avad,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)

	return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):

	assert opt.dataset in ['diem', 'coutrot1', 'coutrot2', 'summe', 'etmd', 'avad']
	print('Creating testing dataset: {}'.format(opt.dataset))

	if opt.dataset == 'diem':
		test_data = saliency_db(
			opt.video_path_diem,
			opt.annotation_path_diem_test,
			opt.salmap_path_diem,
			opt.audio_path_diem,
			spatial_transform,
			temporal_transform,
			target_transform,
			exhaustive_sampling=True)
	elif opt.dataset == 'coutrot1':
		test_data = saliency_db(
			opt.video_path_coutrot1,
			opt.annotation_path_coutrot1_test,
			opt.salmap_path_coutrot1,
			opt.audio_path_coutrot1,
			spatial_transform,
			temporal_transform,
			target_transform,
			exhaustive_sampling=True)
	elif opt.dataset == 'coutrot2':
		test_data = saliency_db(
			opt.video_path_coutrot2,
			opt.annotation_path_coutrot2_test,
			opt.salmap_path_coutrot2,
			opt.audio_path_coutrot2,
			spatial_transform,
			temporal_transform,
			target_transform,
			exhaustive_sampling=True)
	elif opt.dataset == 'summe':
		test_data = saliency_db(
			opt.video_path_summe,
			opt.annotation_path_summe_test,
			opt.salmap_path_summe,
			opt.audio_path_summe,
			spatial_transform,
			temporal_transform,
			target_transform,
			exhaustive_sampling=True)
	elif opt.dataset == 'etmd':
		test_data = saliency_db(
			opt.video_path_etmd,
			opt.annotation_path_etmd_test,
			opt.salmap_path_etmd,
			opt.audio_path_etmd,
			spatial_transform,
			temporal_transform,
			target_transform,
			exhaustive_sampling=True)
	elif opt.dataset == 'avad':
		test_data = saliency_db(
			opt.video_path_avad,
			opt.annotation_path_avad_test,
			opt.salmap_path_avad,
			opt.audio_path_avad,
			spatial_transform,
			temporal_transform,
			target_transform,
			exhaustive_sampling=True)

	return test_data
