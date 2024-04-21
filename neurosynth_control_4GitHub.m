
%  This MATLAB code illustrates the central method in:
%  Luppi, Singleton, Hansen, Jamison, Bzdok, Kuceyeski, Betzel, Misic.
%  "Transitions between cognitive topographies:contributions of network structure,
%  chemoarchitecture, and diagnostic categories." Nature Biomedical Engineering (2024)
%
%  It was developed by Andrea Luppi (al857@cam.ac.uk) in MATLAB 2019a
%
%  To run, ensure you are in the main directory of the repo (i.e. where
%  this file is).
%  Your MATLAB path should also include the Brain Connectivity Toolbox for MATLAB
%  by Rubinov and Sporns (2010) NeuroImage: https://sites.google.com/site/bctnet
%  For additional plotting functionality, also include the ENIGMA Toolbox
%  (https://github.com/MICA-MNI/ENIGMA.git) in your MATLAB path.
%
%  Network control energy depends on the network, but also the start and
%  destination states, and the control strategy.
%  Here we change them one at a time.


clear all;
close all

%%%%%%%%%%%%%%%%%%
%% SET PARAMETERS
%%%%%%%%%%%%%%%%%%

% Information about the atlas that we use;
% adjust accordingly for other atlases:
% Here we use the Desikan-Killiany atlas with 34 cortical regions per
% hemisphere
atlas_info.name = 'DesikanKilliany';
atlas_info.N = 68
atlas_info.atlas_info.cortex_idx_idx = 1:68;

% Parameters for control energy computation
c = 0 % for scaling the SC;(which is then multiplied by the largest eigenvalue)
rho = 1 % balance between minimizing energy or minimizing distance from target state
TimeHorizon = 1 % time horizon for control

% number of rewired networks to generate; the paper uses 500
% but for example purposes here it is set to 10
num_nulls =10 

% Number of parallel cores to use for parfor, since computation can be
% sped up by parallelising with PARFOR;
num_parallel = 4 % adjust as needed

% Start parallel pool to reduce time
% Note that you can also *comment out this code* and replace PARFOR loops
% with regular FOR loops. This may be advisable if trying to use PARFOR is
% causing the script to abort
delete(gcp("nocreate"))
parpool('local', num_parallel)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HOUSEKEEPING 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ensure correct location - you should be working from the base directory
% of the repo, e.g. 'YOUR_OWN_PATH/neurosynth-control'
addpath(genpath(pwd))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load neurosynth maps and terms
load(['data/NeuroSynth_maps_and_terms_', atlas_info.name, num2str(atlas_info.N), '.mat'])
num_terms = size(neurosynth_maps,2);

% Load structural connectome (here we use a consensus connectome from HCP participants)
load(['data/structural_connectome_', atlas_info.name, num2str(atlas_info.N), '.mat'], 'SC')
original_connectome = SC;

% normalize SC to ensure the convergence:
SC = SC ./ (eigs(SC,1, 'largestabs') + c .* eigs(SC,1, 'largestabs') );
A = SC - eye(atlas_info.N);
B = eye(atlas_info.N); % input matrix -- this means that we use all nodes uniformly as controls;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part I: changing the start and destination states
% The core innovation from Luppi et al (2024) NatBME is to use start and target states
% defined as NeuroSynth (https://neurosynth.org/) meta-analytic maps associated
% with different terms from the cognitive neuroscience literature.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Preallocate for speed
transitionEnergyMatrix = zeros(num_terms);

% This is the core of the paper:
% Starting from each NeuroSynth map, how much energy does it take to
% reach each NeuroSynth map?
% The matrix is organised from ROW to COLUMN

for row = 1:num_terms

    disp(['Running task # ', num2str(row), '/', num2str(num_terms)])

    for col = 1:num_terms

        x0 = neurosynth_maps(:,row);
        xT = neurosynth_maps(:,col);

        [ x, u, nodal_A2B ] = fcn_optimalControlContinuous( A, B, rho, x0, xT, TimeHorizon );
        transitionEnergyMatrix(row,col) = mean(nodal_A2B);

    end
end


%% Visualise results

% Useful summaries
energyTO = mean(transitionEnergyMatrix,1);
energyFROM = mean(transitionEnergyMatrix,2)';
overallTransitionEnergy = mean(mean(transitionEnergyMatrix));

% Sort in order of increasing energy to destination, to visualise patterns
[~, OrderTo_all] = sort( energyTO, 'ascend');
fcn_quick_mat_plot(transitionEnergyMatrix(OrderTo_all, OrderTo_all), 'Transition energy (123 NeuroSynth terms)', terms.names(OrderTo_all), terms.names(OrderTo_all))

% Now for the subset of terms
transitionEnergyMatrix_subset = transitionEnergyMatrix(terms.subset_idx, terms.subset_idx);
[~, OrderTo_subset] = sort( mean(transitionEnergyMatrix_subset,1), 'ascend');
fcn_quick_mat_plot(transitionEnergyMatrix_subset(OrderTo_subset, OrderTo_subset), 'Transition energy (25-term subset)', terms.names_subset(OrderTo_subset), terms.names_subset(OrderTo_subset))

% Asymmetry: easier to reach or to leave?
asym = energyTO - energyFROM;
figure; histogram(asym, 40); title({['Asymmetry between energy to reach a state (pos) and energy to leave from that state (neg)']});



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part II: changing the network topology
% Now we can change the network topology, by using different kinds of
% rewiring: degree-preserving (Maslov-Sneppen) rewiring, and a more
% stringent geometry-preserving rewiring that preserves both degree and
% also (binned) connection length, to account for spatial embedding in the brain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Degree-preserving

%Generate nulls
tic
parfor i = 1:num_nulls

    SC_rand = fcn_randmio_und(original_connectome, 1000);

    % normalize SC to ensure the convergence:
    SC_rand = SC_rand ./ (eigs(SC_rand,1, 'largestabs') + c .* eigs(SC_rand,1, 'largestabs') );

    % Store so we can use later
    Amat_rand{i} = SC_rand- eye(atlas_info.N);

end
toc

% Obtain transition energy between each pair of neurosynth maps:
% same as for the empirical connectome above, but needs to be done for each rewired network
transitionEnergy_DegreePreserv = zeros(num_terms, num_terms, num_nulls);
B = eye(atlas_info.N); % input matrix -- this means we use all nodes as controls

tic
parfor i = 1:num_nulls %use parfor because very time-consuming otherwise...
    for row = 1:num_terms

        disp(['Maslov-Sneppen degree-preserving null: Iteration # ', num2str(i), '; Task # ', num2str(row)])

        for col = 1:num_terms

            %Derive Control Energy to transition between overall condition-wide states
            x0 = neurosynth_maps(:,row);
            xT = neurosynth_maps(:,col);

            [ x, u, nodal_A2B ] = fcn_optimalControlContinuous( Amat_rand{i}, B, rho, x0, xT, TimeHorizon );
            transitionEnergy_DegreePreserv(row,col, i) = mean(nodal_A2B);

        end
    end
end
toc


%% Geometry-preserving (spatial embedding)

% Load Euclidean distances for spatial embedding
load(['data/Euclidean_distances_', atlas_info.name, num2str(atlas_info.N), '.mat'], 'Euclidean_dist_mat')

%Generate the nulls
tic
parfor i = 1:num_nulls %parfor to save time

    % Betzel length-preserving rewiring function
    nbins= ceil(atlas_info.N/2);
    [bin,SC_geom] = fcn_match_length_degree_distribution(original_connectome,Euclidean_dist_mat,nbins,20000);

    % function provides only upper triangular so we mirror to lower triangular (since symmetric)
    SC_geom = SC_geom + SC_geom';

    % normalize SC to ensure the convergence:
    SC_geom = SC_geom ./ (eigs(SC_geom,1, 'largestabs') + c .* eigs(SC_geom,1, 'largestabs') );

    %Store so we can use later
    Amat_geom{i} = SC_geom - eye(atlas_info.N);
end
toc

% Obtain transition energy between each pair of neurosynth maps:
% same as for the empirical connectome above, but needs to be done for each rewired network
B = eye(atlas_info.N); % input matrix -- this means we use all nodes as controls
transitionEnergy_GeomPreserv = zeros(num_terms, num_terms, num_nulls);

tic
parfor i = 1:num_nulls
    for row = 1:num_terms

        disp(['Geometric null: Iteration # ', num2str(i), '; Task # ', num2str(row)])

        for col = 1:num_terms

            % Derive Control Energy to transition between overall condition-wide states
            x0 = neurosynth_maps(:,row);
            xT = neurosynth_maps(:,col);

            [ x, u, nodal_A2B ] = fcn_optimalControlContinuous( Amat_geom{i}, B, rho, x0, xT, TimeHorizon );
            transitionEnergy_GeomPreserv(row,col, i) = mean(nodal_A2B);

        end
    end
end
toc

%% Visualise results

% Turn into structures (not allowed within PARFOR)
transitionEnergyNull.GeomPreserv = transitionEnergy_GeomPreserv;
clear transitionEnergy_GeomPreserv
transitionEnergyNull.DegreePreserv = transitionEnergy_DegreePreserv;
clear transitionEnergy_degreePreserv
nulls = fieldnames(transitionEnergyNull);

for nn = 1:numel(nulls)

    for i = 1:size(transitionEnergyNull.(nulls{nn}),3)
        avgTransitionEnergy_null.(nulls{nn})(i) = mean(mean(transitionEnergyNull.(nulls{nn})(:,:,i)));
    end

    % Get min and max for plots
    min_energy_null(nn) = min(avgTransitionEnergy_null.(nulls{nn})(:));
    max_energy_null(nn) = max(avgTransitionEnergy_null.(nulls{nn})(:));

end

min_energy_null(nn+1) =    overallTransitionEnergy;
max_energy_null(nn+1) =    overallTransitionEnergy;

xmin = min(min_energy_null) - 0.01.*min(min_energy_null);
xmax = max(max_energy_null) + 0.01.*max(max_energy_null);

figure;
histogram(avgTransitionEnergy_null.GeomPreserv, 30, 'Normalization', 'probability')
hold on
histogram(avgTransitionEnergy_null.DegreePreserv, 30, 'Normalization', 'probability')
yax=ylim; plot([overallTransitionEnergy, overallTransitionEnergy], [yax(1), yax(2)], 'r', 'LineWidth', 2)
xlim([xmin, xmax])
ylabel('Probability of occurrence')
xlabel(['Energy'])
box off
legend({'Geometry-preserving', 'Degree-preserving'})

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part III: Heterogeneous control
% So far we have kept the B matrix as an identity matrix, meaning
% that we use all nodes uniformly as controls;
% However, this need not be the case.
%
% To use heterogeneous controls, we can add or subtract (or replace) values
% from the identity matrix, for example according to an empirical map of interest;
%
% In this example we use the cortical thickness map.
% Note that depending on your scientific question, you will need to decide
% whether and how to normalise your map.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% In this example we normalise to unit mean so that the overall amount of
% control input is the same, and only the anatomical distribution is
% different between the heterogeneous and uniform case.
load('data/CorticalThickness_DesikanKilliany68.mat', 'CorticalThickness')
InputVector = CorticalThickness ./ mean(CorticalThickness);

% Set up input matrix as heterogeneous
B = diag(InputVector);

A = original_connectome;
A = A ./ (eigs(A,1, 'largestabs') + c .* eigs(A,1, 'largestabs') );
A = A - eye(atlas_info.N);


% Preallocate for speed
transitionEnergyMatrix_heterogeneous = zeros(num_terms);
for row = 1:num_terms
    for col = 1:num_terms

        %Derive Control Energy to transition between overall condition-wide states
        x0 = neurosynth_maps(:,row);
        xT = neurosynth_maps(:,col);

        [ x, u, nodal_A2B ] = fcn_optimalControlContinuous( A, B, rho, x0, xT, TimeHorizon );

        transitionEnergyMatrix_heterogeneous(row,col) = mean(nodal_A2B);
    end
end

%Plot transition landscape with heterogeneous inputs
fcn_quick_mat_plot(transitionEnergyMatrix_heterogeneous(OrderTo_all, OrderTo_all), 'Transition energy with Heterogeneous Control', terms.names(OrderTo_all), terms.names(OrderTo_all))

% Plot correspondence with uniform control
figure;
scatter(transitionEnergyMatrix(:), transitionEnergyMatrix_heterogeneous(:), ...
    'MarkerEdgeColor' , 'k'      , ...
    'MarkerFaceColor' , [.75 .75 .75] )
xlabel('Transition energy with uniform control')
ylabel('Transition energy with heterogeneous control')
