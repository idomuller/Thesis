function palmBW = SegmentPalmFromPick(BW, pick)

%segment palm from second peaks
SE = strel('disk',pick);
palmBW = imopen(BW,SE);

