function newmap = rdbu_sym(h)
%rdbu_sym   Blue, white, and red color map.
%   rdbu_sym(M) returns an M-by-3 matrix containing a blue to white
%   to red colormap, with white corresponding to the CAXIS value closest
%   to zero.  
%   If only negative values are present, only the blue end will be used; if
%   only positive values are present, only the red end will be used.
%
%   This function relies on reds.m and blues_r.m from the ENIGMA Toolbox
%   https://github.com/MICA-MNI/ENIGMA.git

m = size(get(gcf,'colormap'),1);

plottype = gca;
plotID = whos('plottype');
if strcmp(plotID.class, 'matlab.graphics.chart.HeatmapChart')

    % Find middle
    lims = h.ColorLimits;

elseif    strcmp(plotID.class, 'matlab.graphics.axis.Axes')
    % Find middle
    lims = get(gca, 'CLim');

else
    % Find middle
    lims = get(gca, 'CLim');

end

zero = [1 1 1];

pos = Reds;
neg = Blues_r;


% Find whether both pos and neg values are present, or only one
if (lims(1) < 0) & (lims(2) > 0)
    % It has both negative and positive

    if abs(lims(1)) -   abs(lims(2)) < eps
        fullscale = [neg; zero; pos];

    elseif abs(lims(1)) / abs(lims(2)) > 1 %more neg
        fullscale = [neg; zero; pos( 1: round(size(pos,1) * (abs(lims(2)) / abs(lims(1))) ) ,:)];

    elseif abs(lims(1)) / abs(lims(2)) < 1 %more neg
        fullscale = [neg(  round(size(neg,1) * (1-(abs(lims(1)) / abs(lims(2)))) ) : end ,:); zero; pos];

    end

    newmap = fullscale;

elseif lims(1) >= 0

    % Just positive
    newmap = Reds;

else

    % Just negative
    newmap = Blues_r;

end