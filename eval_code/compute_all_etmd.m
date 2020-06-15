function compute_all_etmd(resultspath, annot_base_path)

display('****** Evaluation metrics for ETMD ******')
i=1;
for split=1:3
    %% read database info
    annot_file = [annot_base_path '/fold_lists/ETMD_av_list_test_' num2str(split) '_fps.txt'];
    %
    res_path = [resultspath '/etmd_results_all_split' num2str(split)];
    %
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
    end
    %% evaluate videos
    for k=1:length(test_data)
        vid_res = load(fullfile(res_path, test_data(k).video));
        %% Compute Averages per video
        metrics_avg_video.CC(i) = vid_res.metrics_avg_video.CC;
        metrics_avg_video.similarity(i) = vid_res.metrics_avg_video.similarity;
        metrics_avg_video.NSS(i) = vid_res.metrics_avg_video.NSS;
        metrics_avg_video.AUC_Judd(i) = vid_res.metrics_avg_video.AUC_Judd;
        metrics_avg_video.AUC_shuffled1(i) = vid_res.metrics_avg_video.AUC_shuffled1;
        i=i+1;
    end
end
%% Compute All Averages
metrics_avg_all.CC = mean(metrics_avg_video.CC);
metrics_avg_all.similarity = mean(metrics_avg_video.similarity);
metrics_avg_all.NSS = mean(metrics_avg_video.NSS);
metrics_avg_all.AUC_Judd = mean(metrics_avg_video.AUC_Judd);
metrics_avg_all.AUC_shuffled1 = mean(metrics_avg_video.AUC_shuffled1);
    
metrics_mat = real(cell2mat( struct2cell( metrics_avg_all ) ) )  

fid1=fopen([resultspath, '/', 'final_results_ETMD.txt'],'w');

fprintf(fid1,'****** Evaluation metrics for ETMD ******\n');
fprintf(fid1, 'CC: %0.4f\n', metrics_avg_all.CC);
fprintf(fid1, 'Similarity: %0.4f\n', metrics_avg_all.similarity);
fprintf(fid1, 'NSS: %0.4f\n', metrics_avg_all.NSS);
fprintf(fid1, 'AUC_Judd: %0.4f\n', metrics_avg_all.AUC_Judd);
fprintf(fid1, 'AUC_shuffled: %0.4f\n', metrics_avg_all.AUC_shuffled1);

fclose(fid1);

exit;
