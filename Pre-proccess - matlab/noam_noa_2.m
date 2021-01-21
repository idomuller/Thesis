clc
i=xlsread('C:\Users\ido\Google Drive\Thesis\Data\CSV\No2_csv\Left\No2_L_B_T0.csv');%p222.csv');
i = imrotate(i,90);

% 
N=229;
% SIDE LEFT=1 SIDE RIGHT=0
SIDE=0;
i1=i./max(i(:));
i2 = imresize(i1,4);
imshow(i2)
i3 = imcrop(i2);
i4=imcrop(i2);
i5 = imcrop(i2);
% figure
% imhist(i3)
E_C=entropy(i3);
M_C=max(i(:))*mean2(i3);
K_C=kurtosis(i3(:));
S_C=skewness(i3(:));

E_B=entropy(i4);
M_B=max(i(:))*mean2(i4);
K_B=kurtosis(i4(:));
S_B=skewness(i4(:));

E_P=entropy(i5);
M_P=max(i(:))*mean2(i5);
K_P=kurtosis(i5(:));
S_P=skewness(i5(:));

C = { N SIDE E_C M_C K_C S_C E_B M_B K_B S_B E_P M_P K_P S_P};
% Turn C into a table with relevent header names
writecell(C,'old.xls','Sheet',1,'Range','A9:M9');
%  writecell(C,'middle.xls','Sheet',1,'Range','A9:M9');
% writecell(C,'young.xls','Sheet',1,'Range','A9:M9');



