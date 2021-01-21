load newyorkcity_01012018_01012020.csv;  

data_nyc = newyorkcity_01012018_01012020;

[v_A, d_A, A]= RunOneElecZone(data_nyc, 14, 4, 50, 1);

figure; scatter(d_A(2,2)*v_A(:,2), d_A(3,3)*v_A(:,3), 25,[1:717],'filled')

figure; scatter(d_A(2,2)*v_A(:,2), d_A(4,4)*v_A(:,4), 25,[1:717],'filled')
