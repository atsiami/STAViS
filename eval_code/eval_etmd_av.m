function eval_etmd_av(pred_path, annot_base_path, res_path, videonum ,splitnum)

addpath(genpath('./code_forMetrics'));
i= str2num(videonum);
split = str2num(splitnum);
annot_path = [annot_base_path '/annotations/ETMD_av/'];
%
pred_path = [pred_path '/split' num2str(split) '_results/etmd'];
%
annot_file = [annot_base_path '/fold_lists/ETMD_av_list_test_' num2str(split) '_fps.txt'];
%
res_path = [res_path '/etmd_results_all_split' num2str(split)];
%
mkdir(res_path);
%% read database info
fileID = fopen(annot_file,'r');
txt_data = textscan(fileID,'%s','delimiter','\n'); 
fclose(fileID);
test_data = struct([]);
for ii=1:length(txt_data{1})
    data_split = strsplit(txt_data{1}{ii});
    name = data_split{1};
    nframes  = str2double(data_split{2});
    test_data(ii).video = name;
    test_data(ii).nframes = nframes;
    test_data(ii).annot_path  = fullfile(annot_path, name);
    test_data(ii).pred_path  = fullfile(pred_path, name);
end

%% evaluate videos
file_list = dir([test_data(i).pred_path '/*jpg']);
for j=1:test_data(i).nframes
    eyeMap_all{j} = load(fullfile(test_data(i).annot_path, sprintf('fixMap_%05d.mat',j)));
end
shufMap_all = createShuffmap1(eyeMap_all);
for j = 1:length(file_list)
   tmp = strsplit(file_list(j).name,'.');
   tmp = strsplit(tmp{1},'_');
   frame_num = str2double(tmp{end});
   frame_name = fullfile([test_data(i).pred_path], '', file_list(j).name);
   if (frame_num<=test_data(i).nframes)
       fprintf('video %d of %d: frame %d of %d\n', i, length(test_data), j, length(file_list));
       I_pred = im2double(imread(frame_name));
       I_pred = I_pred(:,:,1);
       I_eye_name = fullfile(test_data(i).annot_path, 'maps', sprintf('eyeMap_%05d.jpg',frame_num));
       I_eye = im2double(imread(I_eye_name));
       I_bin = eyeMap_all{frame_num};
       I_pred_post = postprocessMap(I_pred,2);
       %
       salMap = imresize(I_pred_post,size(I_eye));
       eyeMap = double(I_bin.eyeMap);
       salMap_gt = I_eye;
       shufMap1 = double(shufMap_all);
       shufMap1(eyeMap==1) = 0;
       %% Compute Metrics
       metrics.CC(j) = CC(salMap, salMap_gt);
       metrics.similarity(j) = similarity(salMap, salMap_gt);
       metrics.NSS(j) = NSS(salMap, eyeMap);
       metrics.AUC_Judd(j) = AUC_Judd(salMap, eyeMap);
       metrics.AUC_shuffled1(j) = AUC_shuffled(salMap, eyeMap, shufMap1);
   end
end
%% Compute Averages per video
metrics.CC(isnan(metrics.CC)) = [];
metrics.similarity(isnan(metrics.similarity)) = [];
metrics.NSS(isnan(metrics.NSS)) = [];
metrics.AUC_Judd(isnan(metrics.AUC_Judd)) = [];
metrics.AUC_shuffled1(isnan(metrics.AUC_shuffled1)) = [];
%
metrics_avg_video.CC = mean(metrics.CC);
metrics_avg_video.similarity = mean(metrics.similarity);
metrics_avg_video.NSS = mean(metrics.NSS);
metrics_avg_video.AUC_Judd = mean(metrics.AUC_Judd);
metrics_avg_video.AUC_shuffled1 = mean(metrics.AUC_shuffled1);
%%
save_path = fullfile(res_path, test_data(i).video);
save(save_path, 'metrics', 'metrics_avg_video');
exit;
