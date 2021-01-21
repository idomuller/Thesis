function fingersBW = SegmentFingersFromPicks(BW, picks)

%segment fingers from first peaks
SE = strel('disk',picks(1)-1);
BW3 = imopen(BW,SE);
SE = strel('disk',picks(2));
BW4 = imopen(BW,SE);
fingersBW = BW3 - BW4;
fingersBW = bwareaopen(fingersBW,50);