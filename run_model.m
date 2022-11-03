%% Initialize script

clear all;
clc;

%% Loading data

load('dataset_forHMMGLM_proc.mat');

%% Training selected sub-model of HMM-GLM and obtaining most probable sequence of hidden states via a K-fold CV mechanism

% cross-validation parameters
K = 10;

% model parameters
num_input_dim = D;
num_hidden_states = 6;
num_output_dim = O;
param_init_type = "gaussian_std";
initialize_param = 1;

% training parameters
mode = "baum_welch_auxilary";
method = "gradient_ascent";
num_ep = 100;
return_stats = 0;
inference_method = "forward_algorithm";
inference_form = "log_likelihood";

% variables for storing output
mp_state_seq = zeros(size(labels));
pred_likelihood = zeros(size(labels, 1), num_output_dim, T);
%pred_labels = zeros(size(labels, 1), T); % unnecessary initialization

temp_var_1 = cell(K, 1);
temp_var_2 = cell(K, 1);
t_end_accum = 0;
for k = 1: K
    fprintf('Fold %d\n', k);
    tic;
    
    % training model
    X_train = data(setdiff([1: N_trials], [round((k-1)*(N_trials/K)) + 1: round(k*(N_trials/K))]), :, :);
    Y_train = labels(setdiff([1: N_trials], [round((k-1)*(N_trials/K)) + 1: round(k*(N_trials/K))]), :);
    X_val = data(round((k-1)*(N_trials/K)) + 1: round(k*(N_trials/K)), :, :);
    Y_val = labels(round((k-1)*(N_trials/K)) + 1: round(k*(N_trials/K)), :);
    model = model_HMMGLM(num_input_dim, num_hidden_states, num_output_dim, param_init_type, initialize_param);
    model.train(X_train, Y_train-1, mode, method, num_ep, return_stats, inference_method, inference_form);

    % identifying optimal states in validation sets
    temp_var_3 = zeros(size(Y_val));
    for i = 1: size(X_val, 1)
        temp_var_3(i, :) = model.HSS(squeeze(X_val(i, :, :)), Y_val(i, :)-1);
    end
    temp_var_1{k, 1} = temp_var_3;
    
    % model predictions
    temp_var_4 = zeros(size(Y_val, 1), num_output_dim, T);
    for i = 1: size(X_val, 1)
        temp_var_4(i, :, :) = model.predict_likelihood_onestep(squeeze(X_val(i, :, :)), Y_val(i, :));
    end
    temp_var_2{k, 1} = temp_var_4;
    
    t_end = toc;
    t_end_accum = t_end_accum + t_end;
    fprintf("Time elapsed: %f \n", t_end);
    fprintf("Total time elapsed: %f \n", t_end_accum);
end

for k = 1: K
    mp_state_seq(round((k-1)*(N_trials/K)) + 1: round(k*(N_trials/K)), :) = temp_var_1{k, 1};
    pred_likelihood(round((k-1)*(N_trials/K)) + 1: round(k*(N_trials/K)), :, :) = temp_var_2{k, 1};
end
[~, pred_labels] =max(pred_likelihood, [], 2);
pred_labels = squeeze(pred_labels);

clearvars unique_animal_ids K num_input_dim num_hidden_states
clearvars num_output_dim param_init_type initialize_param mode method
clearvars num_ep return_stats inference_method inference_form temp_var_1
clearvars temp_var_2 t_end_accum k X_train Y_train X_val Y_val model
clearvars temp_var_3 i temp_var_4 t_end temp_var_5 pred_label_local

%% Computing AUC scores for the ROC and PR curves from predicted likelihoods

% Running parameters
K = 10;

% variables for storing output
auc_ROC = zeros(O, K);
auc_ROC_delay = zeros(O, K);

for k = 1: K
    % representing the labels using one-hot encoding for one-vs-all performance
    % curve analysis
    labels_local = labels(round((k-1)*(N_trials/K)) + 1: round(k*(N_trials/K)), :);
    pred_likelihood_local = pred_likelihood(round((k-1)*(N_trials/K)) + 1: round(k*(N_trials/K)), :, :);
    labels_oh = zeros(O, numel(labels_local)); % one-hot encoding
    labels_oh_delay = zeros(O, numel(labels_local) - size(labels_local, 1)); % one-hot encoding with delay
    for o = 1: O
        labels_oh(o, :) = (reshape(labels_local, 1, numel(labels_local))==o);
        labels_oh_delay(o, :) = (reshape(labels_local(:, 1: T-1), 1, numel(labels_local) - size(labels_local, 1))==o);
    end

    % computing auc scores
    for o = 1: O
        [~, ~, ~, auc_local] = perfcurve(labels_oh(o, :), reshape(pred_likelihood_local(:, o, :), 1, numel(pred_likelihood_local(:, o, :))), 1);
        auc_ROC(o, k) = auc_local;
        [~, ~, ~, auc_local] = perfcurve(labels_oh_delay(o, :), reshape(pred_likelihood_local(:, o, 2: T), 1, numel(pred_likelihood_local(:, o, 2: T))), 1);
        auc_ROC_delay(o, k) = auc_local;
    end
end

clearvars K k labels_local pred_likelihood_local labels_oh labels_oh_delay
clearvars o auc_local