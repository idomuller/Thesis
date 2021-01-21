jpgImage = imread('C:\Users\ido\Google Drive\Thesis\Data\JPG-Recover\1_Nir\right\ניר - ימין\flir_20200629T123959.jpg');

xlsImage = xlsread('C:\Users\ido\Google Drive\Thesis\Data\CSV\No1_CSV\right\No1_R_B_T0.csv');

figure(23);
subplot(1,2,1);
imagesc(jpgImage)
subplot(1,2,2);
imagesc(xlsImage);
x=1;