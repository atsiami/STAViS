function [salMap] = postprocessMap(netMap,sigma)

xMin=0;
xMax=255;
salMap = squeeze(netMap);
salMap = 255*salMap;
salMap(salMap>xMax) = xMax;
salMap(salMap<xMin) = xMin;
salMap = (salMap / max(salMap(:)))*255;

salMap = normalizeImage1(salMap);
h_c=fspecial('gaussian',2*ceil(3*sigma)+1,sigma);
salMap =  imfilter(double(salMap),h_c,'symmetric','conv');