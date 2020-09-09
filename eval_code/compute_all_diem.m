function compute_all_diem(resultspath, annot_base_path) 

display('****** Evaluation metrics for DIEM ******')
i=1;
for split=1:3
    annot_file = [annot_base_path '/fold_lists/DIEM_list_test_fps.txt'];
    %
    res_path = [resultspath '/diem_results_all_split' num2str(split)];
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
    i=1;
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
    %% Compute All Averages
    metrics_avg_all.CC = mean(metrics_avg_video.CC);
    metrics_avg_all.similarity = mean(metrics_avg_video.similarity);
    metrics_avg_all.NSS = mean(metrics_avg_video.NSS);
    metrics_avg_all.AUC_Judd = mean(metrics_avg_video.AUC_Judd);
    metrics_avg_all.AUC_shuffled1 = mean(metrics_avg_video.AUC_shuffled1);

    metrics_mat(:,split) = real(cell2mat( struct2cell( metrics_avg_all ) ) );
end

metrics_mat1 = mean(metrics_mat,2)

fid1=fopen([resultspath, '/', 'final_results_DIEM.txt'],'w');

fprintf(fid1,'****** Evaluation metrics for DIEM ******\n');
fprintf(fid1, 'CC: %0.4f\n', metrics_mat1(1));
fprintf(fid1, 'Similarity: %0.4f\n', metrics_mat1(2));
fprintf(fid1, 'NSS: %0.4f\n', metrics_mat1(3));
fprintf(fid1, 'AUC_Judd: %0.4f\n', metrics_mat1(4));
fprintf(fid1, 'AUC_shuffled: %0.4f\n', metrics_mat1(5));

fclose(fid1);

exit;
