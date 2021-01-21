up = envelope(dy,1,'peak');
plot(radii(2:end,1),up,radii(2:end,1),dy,'linewidth',1.5)
legend('up','dy')

up = envelope(dy,1,'peak');
plot(radii(1:48,1),up(1:48,1),radii(1:48,1),dy(1:48,1),'linewidth',1.5)
legend('up','dy')



[peaks,locations] = findpeaks(up(1:48,1),radii(1:48,1));
pks = [peaks,locations];