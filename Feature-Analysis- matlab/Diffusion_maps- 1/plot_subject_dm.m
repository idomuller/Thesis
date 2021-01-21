load ZB_distal_left.csv
load ZB_proxy_left.csv
load ZB_distal_right.csv
load ZB_proxy_right.csv


figure; plot(median(ZB_distal_left'), 'b')
hold on;
 plot(median(ZB_proxy_left'), 'r')
hold on;
plot(median(ZB_distal_right'), 'c')
hold on;
plot(median(ZB_proxy_right'), 'm')

load LM_distal_left.csv
load LM_proxy_left.csv
load LM_distal_right.csv
load LM_proxy_right.csv

figure; plot(median(LM_distal_left'), 'b')
hold on;
plot(median(LM_proxy_left'), 'r')
hold on;
plot(median(LM_distal_right'), 'c')
hold on;
plot(median(LM_proxy_right'), 'm')


% all of ZB's data
ZB = [ZB_distal_left;ZB_proxy_left;ZB_distal_right;ZB_proxy_right];

% all of LM's data
LM = [LM_distal_left;LM_proxy_left;LM_distal_right;LM_proxy_right];

dist = squareform(pdist([ZB;LM]));
[ U, d_A, v_A, sigma ] = AnalyzeGraphAlpha(  dist,1, 'max', 10, 10);

lbl_ZB_LM = zeros(168,1);
lbl_ZB_LM(85:end)=1;

% ZB in blue, LM in yellow
figure; scatter(v_A(:,2), v_A(:,3), 20, lbl_ZB_LM, 'filled')

ZB_distal_left_n = ZB_distal_left./norm(ZB_distal_left);
ZB_proxy_left_n = ZB_proxy_left./norm(ZB_proxy_left);
ZB_distal_right_n = ZB_distal_right./norm(ZB_distal_right);
ZB_proxy_right_n = ZB_proxy_right./norm(ZB_proxy_right);

LM_distal_left_n = LM_distal_left./norm(LM_distal_left);
LM_proxy_left_n = LM_proxy_left./norm(LM_proxy_left);
LM_distal_right_n = LM_distal_right./norm(LM_distal_right);
LM_proxy_right_n = LM_proxy_right./norm(LM_proxy_right);

% all of ZB's normalized data
ZBn = [ZB_distal_left_n;ZB_proxy_left_n;ZB_distal_right_n;ZB_proxy_right_n];

% all of LM's data
LMn = [LM_distal_left_n;LM_proxy_left_n;LM_distal_right_n;LM_proxy_right_n];


distn = squareform(pdist([ZBn;LMn]));
[ U, d_An, v_An, sigma ] = AnalyzeGraphAlpha(  distn,1, 'max', 10, 10);


% ZB in blue, LM in yellow
figure; scatter(v_An(:,2), v_An(:,3), 20, lbl_ZB_LM, 'filled')

lbl_by_area_t = zeros(84,1);  % dark blue - left hand distal
lbl_by_area_t(21*1+1:21*2)=10; % bright yellow - left hand proxy
lbl_by_area_t(21*2+1:21*3)=3; % bright yellow - left hand proxy
lbl_by_area_t(21*3+1:21*4)=7;

lbl_by_area = [lbl_by_area_t,lbl_by_area_t];
figure; scatter(v_An(1:84,2), v_An(1:84,3), 20, lbl_by_area_t, 'filled')
figure; scatter(v_An(85:168,2), v_An(85:168,3), 20, lbl_by_area_t, 'filled')
