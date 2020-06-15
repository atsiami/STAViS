% normalizeImage - linearly normalize an array.
%
% res = normalizeImage(data);
%    Linearly normalize data between 0 and 1.
%
% res = normalizeImage(img,range);
%    Lineary normalize data between range(1) and range(2)
%    instead of [0,1]. The special value range = [0 0]
%    means that no normalization is performed, i.e., res = img.

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2007
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.TXT document for the license agreement.
% More information about this project is available at:
% http://www.saliencytoolbox.net

function res = normalizeImage1(img,varargin)

if (length(varargin) >= 1),
    range = varargin{1};
else range = [0,1];
end

if ((range(1) == 0) && (range(2) == 0))
    res = img;
    return;
end

mx = max(img(:));
mn = min(img(:));

if (mx == mn)
    res = img - mx + 0.5*sum(range);
else
    res = (img - mn) / (mx - mn) * abs(range(2)-range(1)) + min(range);
end