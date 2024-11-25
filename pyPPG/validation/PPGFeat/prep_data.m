clear all
close all

% Load test referece data
test_date='2024_1_7_16_1';
input_folder=['..',filesep,'results',filesep,test_date,filesep,'MG_PC',filesep];
load([input_folder,'MG_PC.mat'])
input=['..',filesep,'PPG-BP_annot',filesep,'PPG-BP_ref1.mat'];
load(input)

% Load PPG data
input=['..',filesep,'PPG-BP_annot',filesep,'PPG-BP_ref1.mat'];
load(input)

all_ppg=[];
for i=1:length(ppg_data)
    tmp_sig=ppg_data(i).sig';
    all_ppg(i,1:2100)=tmp_sig(1:2100);
end

% Create new PPG data for PPGfeat benchmarking
new_ppg=[];
start_sig=[];
for i=1:length(ppg_data)
    delta_t1=100;
    start_s=MG_fps(i).on-delta_t1;
    if start_s<20
        start_s=1;
    end

    start_sig(i,1)=start_s;
    end_s=MG_fps(i).off+delta_t1;
    delta_t2=50;

    tmp1_ppg=all_ppg(i,start_s:end_s);
    tmp2_ppg=tmp1_ppg(delta_t1+delta_t2:end)+tmp1_ppg(end)-tmp1_ppg(delta_t1+delta_t2);
    tmp3_ppg=[tmp1_ppg,tmp2_ppg];
    tmp4_ppg=tmp1_ppg(delta_t1+delta_t2:end)+tmp3_ppg(end)-tmp3_ppg(delta_t1+delta_t2);
    tmp5_ppg=[tmp3_ppg,tmp4_ppg];
    tmp6_ppg=tmp5_ppg(1:1800);
    new_ppg(i,:)=tmp6_ppg;
end

% Save new PPG signal and starting points
writematrix(start_sig, 'start_sig.csv');
writematrix(new_ppg, 'PPGdata.csv');