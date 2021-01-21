%---------------------------Extract Points--------------------------------%
%extract all points which fall on the thumb line
m = thumb(1,5);
c = thumb(1,6);

thumb_points = [0,0];
thumb_steps = (max(thumb(1,1),thumb(1,3))-min(thumb(1,1),thumb(1,3)))/100;
for i = min(thumb(1,2),thumb(1,4)):max(thumb(1,2),thumb(1,4))
    y = i;
    for j = min(thumb(1,1),thumb(1,3)):thumb_steps:max(thumb(1,1),thumb(1,3))
        if y == round((m*j)+(c))
            thumb_points = [thumb_points;j,y];
        end
    end
end
thumb_points = thumb_points(2:end,:);

%extract all points which fall on the index line
m = index(1,5);
c = index(1,7);

index_points = [0,0];
index_steps = (max(index(1,1),index(1,3))-min(index(1,1),index(1,3)))/100;
for i = min(index(1,2),index(1,4)):max(index(1,2),index(1,4))
    y = i;
    for j = min(index(1,1),index(1,3)):index_steps:max(index(1,1),index(1,3))
        if y == round((m*j)+(c))
            index_points = [index_points;j,y];
        end
    end
end
index_points = index_points(2:end,:);

%extract all points which fall on the little line
m = little(1,5);
c = little(1,7);

little_points = [0,0];
little_steps = (max(little(1,1),little(1,3))-min(little(1,1),little(1,3)))/100;
for i = min(little(1,2),little(1,4)):max(little(1,2),little(1,4))
    y = i;
    for j = min(little(1,1),little(1,3)):little_steps:max(little(1,1),little(1,3))
        if y == round((m*j)+(c))
            little_points = [little_points;j,y];
        end
    end
end
little_points = little_points(2:end,:);
