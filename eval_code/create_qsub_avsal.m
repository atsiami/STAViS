function create_qsub_avsal(res_path, annot_base_path, pred_path)



%%%% DIEM %%%%%%
fid1=fopen(['eval_code/DIEM_eval.sh'],'w');
fprintf(fid1,'#!/usr/bin/env bash\n\n');
for split=1:3
    for i=1:17
        fprintf(fid1,'matlab -nodesktop -nodisplay -nojvm ');
        fprintf(fid1,['-r "addpath(genpath(' char(39) './eval_code' char(39) ')); ']);
        fprintf(fid1,'eval_diem(''%s'', ''%s'', ''%s'', ''%s'', ''%s'')\"\n', pred_path, annot_base_path, res_path, num2str(i),num2str(split));
    end
end
fclose(fid1);


%%%%%% ETMD %%%%%%%
for split=1:3
    annot_file = [annot_base_path '/fold_lists/ETMD_av_list_test_' num2str(split) '_fps.txt'];
    fileID = fopen(annot_file,'r');
    txt_data = textscan(fileID,'%s','delimiter','\n');
    fclose(fileID);
    vid_nums(split)=length(txt_data{1});
end
fid1=fopen(['eval_code/ETMD_eval.sh'],'w');
fprintf(fid1,'#!/usr/bin/env bash\n\n');
for split=1:3
    for i=1:vid_nums(split)
        fprintf(fid1,'matlab -nodesktop -nodisplay -nojvm ');
        fprintf(fid1,['-r "addpath(genpath(' char(39) './eval_code' char(39) ')); ']);
        fprintf(fid1,'eval_etmd_av(''%s'', ''%s'', ''%s'', ''%s'', ''%s'')\"\n', pred_path, annot_base_path, res_path, num2str(i),num2str(split));
    end
end
fclose(fid1);


%%%% Coutrot db1 %%%%%%%%%
for split=1:3
    annot_file = [annot_base_path '/fold_lists/Coutrot_db1_list_test_' num2str(split) '_fps.txt'];
    fileID = fopen(annot_file,'r');
    txt_data = textscan(fileID,'%s','delimiter','\n');
    fclose(fileID);
    vid_nums(split)=length(txt_data{1});
end

fid1=fopen(['eval_code/Coutrot1_eval.sh'],'w');
fprintf(fid1,'#!/usr/bin/env bash\n\n');
for split=1:3
    for i=1:vid_nums(split)
        fprintf(fid1,'matlab -nodesktop -nodisplay -nojvm ');
        fprintf(fid1,['-r "addpath(genpath(' char(39) './eval_code' char(39) ')); ']);
        fprintf(fid1,'eval_coutrot1(''%s'', ''%s'', ''%s'', ''%s'', ''%s'')\"\n', pred_path, annot_base_path, res_path, num2str(i),num2str(split));
    end
end
fclose(fid1);

%%%%% Coutrot db2 %%%%%%%%
for split=1:3
    annot_file = [annot_base_path '/fold_lists/Coutrot_db2_list_test_' num2str(split) '_fps.txt'];
    fileID = fopen(annot_file,'r');
    txt_data = textscan(fileID,'%s','delimiter','\n');
    fclose(fileID);
    vid_nums(split)=length(txt_data{1});
end
fid1=fopen(['eval_code/Coutrot2_eval.sh'],'w');
fprintf(fid1,'#!/usr/bin/env bash\n\n');
for split=1:3
    for i=1:vid_nums(split)
        fprintf(fid1,'matlab -nodesktop -nodisplay -nojvm ');
        fprintf(fid1,['-r "addpath(genpath(' char(39) './eval_code' char(39) ')); ']);
        fprintf(fid1,'eval_coutrot2(''%s'', ''%s'', ''%s'', ''%s'', ''%s'')\"\n', pred_path, annot_base_path, res_path, num2str(i),num2str(split));
    end
end
fclose(fid1);

%%%%% SumMe %%%%%%%
for split=1:3
    annot_file = [annot_base_path '/fold_lists/SumMe_list_test_' num2str(split) '_fps.txt'];
    fileID = fopen(annot_file,'r');
    txt_data = textscan(fileID,'%s','delimiter','\n');
    fclose(fileID);
    vid_nums(split)=length(txt_data{1});
end
fid1=fopen(['eval_code/SumMe_eval.sh'],'w');
fprintf(fid1,'#!/usr/bin/env bash\n\n');
for split=1:3
    for i=1:vid_nums(split)
        fprintf(fid1,'matlab -nodesktop -nodisplay -nojvm ');
        fprintf(fid1,['-r "addpath(genpath(' char(39) './eval_code' char(39) ')); ']);
        fprintf(fid1,'eval_summe(''%s'', ''%s'', ''%s'', ''%s'', ''%s'')\"\n', pred_path, annot_base_path, res_path, num2str(i),num2str(split));
    end
end
fclose(fid1);


%%%%% AVAD %%%%%%%
for split=1:3
    annot_file = [annot_base_path '/fold_lists/AVAD_list_test_' num2str(split) '_fps.txt'];
    fileID = fopen(annot_file,'r');
    txt_data = textscan(fileID,'%s','delimiter','\n');
    fclose(fileID);
    vid_nums(split)=length(txt_data{1});
end
fid1=fopen(['eval_code/AVAD_eval.sh'],'w');
fprintf(fid1,'#!/usr/bin/env bash\n\n');
for split=1:3
    for i=1:vid_nums(split)
        fprintf(fid1,'matlab -nodesktop -nodisplay -nojvm ');
        fprintf(fid1,['-r "addpath(genpath(' char(39) './eval_code' char(39) ')); ']);
        fprintf(fid1,'eval_avad(''%s'', ''%s'', ''%s'', ''%s'', ''%s'')\"\n', pred_path, annot_base_path, res_path, num2str(i),num2str(split));
    end
end
fclose(fid1);

exit;
end
