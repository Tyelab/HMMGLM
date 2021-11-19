classdef model_HMMGLM < handle
    
    properties %variables
        F_transition;
        F_emission;
        Pi;
        num_input_dim;
        num_hidden;
        num_output;
        param_init_type;
        param_init_param;
        num_param;
    end
    
    methods
        
        % Constructor
        function obj = model_HMMGLM(num_input_dim, num_hidden, num_output, param_init_type, initialize_param)
            % Purpose: To initialize the model
            %
            % Input:
            % num_input_dim: positive integer
            %     the dimensionality of the feature vector representing the
            %     evidence at any time stamp
            % num_hidden: positive_integer
            %     the number of hidden states to be assumed for the model
            % num_output: positive integer
            %     the number of discrete labels that a time point can be
            %     assigned
            % param_init_type: string belonging to {"gaussian_std"}
            %     depicting the type of initialization to be used for the
            %     parameters
            % initialize_param: boolean
            %     whether or not to initialize the parameters at the time
            %     of creation of the model
            %
            % Output: 
            % <object>: object of class model_HMM
            %     model_HMMGLM object with the specifications provided
            %     with the constructor
            
            % Code:
            obj.num_input_dim = num_input_dim;
            obj.num_hidden = num_hidden;
            obj.num_output = num_output;
            obj.F_transition = zeros(num_input_dim+1, num_hidden, num_hidden);
            obj.F_emission = zeros(num_input_dim+1, num_hidden, num_output);
            obj.Pi = zeros(num_hidden, 1);
            obj.num_param = size(obj.F_emission(:), 1) + size(obj.F_transition(:), 1) + size(obj.Pi(:), 1);
            obj.param_init_type = param_init_type;
            if initialize_param
                obj.parameter_initialization();
            end
        end
        
        % Parameter Initialization
        function parameter_initialization(obj)
            % Purpose: To initialize the parameters of the model
            %
            % Input:
            % <none>
            %
            % Output:
            % <none>
            
            % Code:
            num_param = size(obj.F_transition(:), 1) + size(obj.F_emission(:), 1) + size(obj.Pi(:), 1);
            obj.param_init_param = struct();
            if obj.param_init_type == "gaussian_std"
                obj.param_init_param.mu = 0;
                obj.param_init_param.sigma_sq = 0.001;
                init_param = normrnd(obj.param_init_param.mu, obj.param_init_param.sigma_sq, num_param, 1);
                obj.F_transition(:) = init_param(1: size(obj.F_transition(:), 1), 1);
                obj.F_emission(:) = init_param(size(obj.F_transition(:), 1) + 1: size(obj.F_transition(:), 1) + size(obj.F_emission(:), 1), 1);
                obj.Pi(:) = init_param(size(obj.F_transition(:), 1) + size(obj.F_emission(:), 1) + 1: num_param, 1);
                obj.Pi = obj.Pi/sum(obj.Pi, 'all');
            end
            obj.Pi(:) = (1/size(obj.Pi(:), 1))*ones(size(obj.Pi, 1), 1);
            obj.Pi = (1/sum(obj.Pi, 'all'))*obj.Pi;
        end
        
        % Training
        function [inference_out, val_inference_out] = train(obj, X, Y, mode, method, num_ep, return_stats, varargin)
            % Purpose: To fit the training data to the model and estimate
            %          the parameters
            %
            % Input: 
            % X: array with dimensions num_input_seq*num_input_dim*time
            %     the evidence sequences for various trials across time
            % Y: array with dimensions num_input_seq*time
            %     depicting the label configuration sequences for various
            %     trials across time
            % mode: string belonging to {"baum_welch_auxilary"};
            %     objective function to be used for optimization;
            %     baum_welch_auxilary = the baum-welch auxilary function
            % method: string belonging to {"gradient_ascent", "newton"}
            %     training algorithm to be used to optimize the objective
            %     function; "gradient_ascent" = gradient ascent algorithm,
            %     "newton" = newton's method which computes the hessian
            %     (EXPERIMENTAL)
            % num_ep: positive integer
            %     the number of epochs for which to run the learning
            %     algorithm
            % return_stats: boolean
            %     whether or not to return the inference statistics for the
            %     training data during the fitting
            %
            % Optional Input:
            % inference_method: string belonging to {"forward_algorithm"}
            %     the method to be used for inference
            % inference_form: string belonging to {"likelihood", 
            %                 "log_likelihood"}
            %     the form in which to report the inference output
            % val: boolean
            %     whether to track inference on a provided validation set instead
            % X_val: array with dimensions num_input_seq*num_input_dim*time
            %     the evidence sequences for various trials across time
            % Y_val: array with dimensions num_input_seq*time array
            %     the label configuration sequences for various trials
            %     across time
            %
            % Output:
            % inference_out: array with dimensions num_ep*1
            %     the inference outputs for each step of training; if
            %     return_stats is set to zero, this would be a zero array
            
            % Code:
            
            % optional args
            nVarargs = length(varargin);
            if nVarargs > 0 && nVarargs < 3
                inference_method = varargin{1};
                inference_form = varargin{2};
            end
            if nVarargs > 3
                inference_method = varargin{1};
                inference_form = varargin{2};
                val = varargin{3};
                X_val = varargin{4};
                Y_val = varargin{5};
            else
                val = 0;
            end
            
            M = size(Y, 1);
            K = obj.num_hidden;
            J = obj.num_output;
            T = size(X, 3);
            inference_out = zeros(num_ep, 1);
            if val
                val_inference_out = zeros(num_ep, 1);
            end
            
            if mode == "baum_welch_auxilary"
                if method == "gradient_ascent"
                    % setting learning rate annealing
                    eta_0 = 10^(-6);
                    Temp = 128;
                    % learning
                    ep = 0;
                    while ep < num_ep

                        % learning rate
                        eta_ep = eta_0/(1 + (ep/Temp));

                        % accessing current parameter values
                        F_tr = obj.F_transition;
                        F_em = obj.F_emission;
                        Pi = obj.Pi;

                        % computing gradients
                        grad_F_tr = zeros(size(F_tr));
                        grad_F_em = zeros(size(F_em));
                        grad_Pi = zeros(size(Pi));
                        parfor m = 1: M
                            X_m = squeeze(X(m, :, :));
                            Y_m = squeeze(Y(m, :));
                            alpha_m = obj.alpha(X_m);
                            eta_m = obj.eta(X_m);
                            [a_t_m, c_t_m] = obj.forward(X_m, Y_m, alpha_m, eta_m, 1, 1);
                            b_t_m = obj.backward(X_m, Y_m, alpha_m, eta_m, 1, c_t_m);
                            temp_var = obj.gamma(a_t_m, b_t_m);
                            gamma_m = obj.gamma(a_t_m, b_t_m);
                            zeta_m = obj.zeta(Y_m, a_t_m, b_t_m, alpha_m, eta_m);
                            grad_Pi_local = zeros(size(Pi));
                            grad_F_tr_local = zeros(size(F_tr));
                            grad_F_em_local = zeros(size(F_em));
                            for r = 1: K
                                grad_Pi_local(r, 1) = grad_Pi_local(r, 1) + gamma_m(r, 1)/Pi(r, 1);
                                for t = 1: T
                                    if t > 1
                                        for s = 1: K
                                            grad_F_tr_local(:, r, s) = grad_F_tr_local(:, r, s) + (zeta_m(r, s, t) - alpha_m(r, s, t)*gamma_m(r, t-1))*vertcat(X_m(:, t), 1);
                                        end
                                    end
                                    for l = 1: J
                                        grad_F_em_local(:, r, l) = grad_F_em_local(:, r, l) + ((Y_m(1, t)+1==l) - eta_m(r, l, t))*gamma_m(r, t)*vertcat(X_m(:, t), 1);
                                    end
                                end
                            end
                            grad_Pi = grad_Pi + grad_Pi_local;
                            grad_F_tr = grad_F_tr + grad_F_tr_local;
                            grad_F_em = grad_F_em + grad_F_em_local;
                        end
                        
                        % updating number of epochs passed
                        ep = ep + 1;
                        
                        % adding priors to gradients
                        if obj.param_init_type == "gaussian_std"
                            sigma_sq = obj.param_init_param.sigma_sq;
                            grad_F_tr = grad_F_tr - (1/sigma_sq)*F_tr;
                            grad_F_em = grad_F_em - (1/sigma_sq)*F_em;
                        end
                        
                        % updating parameters
                        obj.Pi = obj.Pi + eta_ep*grad_Pi;
                        obj.F_transition = obj.F_transition + eta_ep*grad_F_tr;
                        obj.F_emission = obj.F_emission + eta_ep*grad_F_em;
                        
                        % applying requisite constraints
                        obj.Pi = obj.Pi/sum(obj.Pi, 'all');
                        
                        % computing likelihood of training set
                        if return_stats
                            inference_out(ep, 1) = obj.inference(X, Y, inference_method, inference_form);
                            if val
                                val_inference_out(ep, 1) = obj.inference(X_val, Y_val, inference_method, inference_form);
                            end
                        end
                    end
                elseif method == "newton" % EXPERIMENTAL -- NEEDS MODIFICATIONS
                    %learning
                    ep = 0;
                    while ep < num_ep
                        
                        for m = 1: M
                            % accessing current parameter values
                            F_tr = obj.F_transition;
                            F_em = obj.F_emission;
                            Pi = obj.Pi;
                            
                            %computing steps
                            step_F_tr = zeros(size(F_tr));
                            step_F_em = zeros(size(F_em));
                            step_Pi = zeros(size(Pi));
                            
                            X_m = squeeze(X(m, :, :));
                            Y_m = squeeze(Y(m, :));
                            alpha_m = obj.alpha(X_m);
                            eta_m = obj.eta(X_m);
                            [a_t_m, c_t_m] = obj.forward(X_m, Y_m, alpha_m, eta_m, 1, 1);
                            b_t_m = obj.backward(X_m, Y_m, alpha_m, eta_m, 1, c_t_m);
                            gamma_m = obj.gamma(a_t_m, b_t_m);
                            zeta_m = obj.zeta(Y_m, a_t_m, b_t_m, alpha_m, eta_m);
                            epsilon = 10^(-8);
                            
                            for r = 1: K
                                grad_Pi_temp = gamma_m(r, 1)/Pi(r, 1);
                                hess_Pi_temp = gamma_m(r, 1)/(Pi(r, 1)^2);
                                step_Pi(r, 1) = step_Pi(r, 1) + inv(hess_Pi_temp + epsilon*eye(size(hess_Pi_temp)))*grad_Pi_temp;
                                for t = 1: T
                                    if t > 1
                                        for s = 1: K
                                            grad_F_tr_temp = (zeta_m(r, s, t) - alpha_m(r, s, t)*gamma_m(r, t-1))*vertcat(X_m(:, t), 1);
                                            hess_F_tr_temp = -(1-alpha_m(r, s, t))*alpha_m(r, s, t)*gamma_m(r, t-1)*(vertcat(X_m(:, t), 1)*vertcat(X_m(:, t), 1)');
                                            if obj.param_init_type == "gaussian_std"
                                                sigma_sq = obj.param_init_param.sigma_sq;
                                                grad_F_tr_temp = grad_F_tr_temp - (1/sigma_sq)*F_tr(:, r, s);
                                                hess_F_tr_temp = hess_F_tr_temp - (1/sigma_sq)*eye(size(hess_F_tr_temp));
                                            end
                                            step_F_tr(:, r, s) = step_F_tr(:, r, s) + inv(hess_F_tr_temp + epsilon*eye(size(hess_F_tr_temp)))*grad_F_tr_temp;
                                        end
                                    end
                                    for l = 1: J
                                        grad_F_em_temp = ((Y_m(1, t)+1==l) - eta_m(r, l, t))*gamma_m(r, t)*vertcat(X_m(:, t), 1);
                                        hess_F_em_temp = -(1-eta_m(r, l, t))*eta_m(r, l, t)*gamma_m(r, t)*(vertcat(X_m(:, t), 1)*vertcat(X_m(:, t), 1)');
                                        if obj.param_init_type == "gaussian_std"
                                                sigma_sq = obj.param_init_param.sigma_sq;
                                                grad_F_em_temp = grad_F_em_temp - (1/sigma_sq)*F_em(:, r, l);
                                                hess_F_em_temp = hess_F_em_temp - (1/sigma_sq)*eye(size(hess_F_em_temp));
                                        end
                                        step_F_em(:, r, l) = step_F_em(:, r, l) + inv(hess_F_em_temp + epsilon*eye(size(hess_F_em_temp)))*grad_F_em_temp;
                                    end
                                end
                            end
                            
                            % updating parameters
                            obj.Pi = obj.Pi + step_Pi;
                            obj.F_transition = obj.F_transition + step_F_tr;
                            obj.F_emission = obj.F_emission + step_F_em;
                            
                            % applying requisite constraints
                            obj.Pi = obj.Pi/sum(obj.Pi, 'all');
                        end
                        
                        % updating number of epochs passed
                        ep = ep + 1;

                        % computing likelihood of training set
                        if return_stats
                            inference_out(ep, 1) = obj.inference(X, Y, inference_method, inference_form);
                            if val
                                val_inference_out(ep, 1) = obj.inference(X_val, Y_val, inference_method, inference_form);
                            end
                        end
                    end
                end
            end
        end
        
        % Inference
        function inference_out = inference(obj, X, Y, inference_method, inference_form)
            % Purpose: To compute a measure of the likelihood of the given
            %          data provided the model parameters
            %
            % Input: 
            % X: array with dimensions num_input_seq*num_input_dim*time
            %     the evidence sequences for various trials across time
            % Y: array with dimensions num_input_seq*time
            %     depicting the label configuration sequences for various
            %     trials across time
            %
            % Optional Input:
            % inference_method: string belonging to {"forward_algorithm"}
            %     the method to be used for inference
            % inference_form: string belonging to {"likelihood", 
            %                 "log_likelihood"}
            %     the form in which to report the inference output
            %
            % Output:
            % inference_out: double
            %   measure of inference in the requested format
            
            % Code:
            inference_out = 0;
            M = size(Y, 1);
            T = size(Y, 2);
            if inference_method == "forward_algorithm"
                for m = 1: M
                    X_m = squeeze(X(m, :, :));
                    Y_m = squeeze(Y(m, :, :));
                    if m == 17
                        "hey";
                    end
                    alpha_m = obj.alpha(X_m);
                    eta_m = obj.eta(X_m);
                    [~, c_t_m] = obj.forward(X_m, Y_m, alpha_m, eta_m, 1, 1);
                    if inference_form == "likelihood"
                        inference_out = inference_out + 1/(prod(c_t_m));
                    elseif inference_form == "log_likelihood"
                        inference_out = inference_out - sum(log(c_t_m));
                    end
                end
            end
        end 
        
        % Prediction
        function Y_pred = predict_circular(obj, X, Y)
            % Purpose: To predict the behavior labels for a given data
            %
            % Input: 
            % X: array with dimensions num_input_seq*num_input_dim*time
            %     the evidence sequences for various trials across time
            % Y: array with dimensions num_input_seq*time
            %     depicting the label configuration sequences for various
            %     trials across time
            %
            % Output:
            % Y_pred: array with dimensions num_input_seq*time
            %     the predicted label configuration sequences for various
            %     trials across time
            
            % Code:
            state_seq = zeros(size(Y));
            for i = 1: size(X, 1)
                state_seq(i, :) = obj.HSS(squeeze(X(i, :, :)), squeeze(Y(i, :)));
            end
            Y_pred = zeros(size(Y));
            for i = 1: size(X, 1)
                for j = 1: size(X, 3)
                    sample = [squeeze(X(i, :, j)), 1];
                    temp_val = sample*squeeze(obj.F_emission(:, state_seq(i, j), :));
                    temp_prob = exp(temp_val)/sum(exp(temp_val), 'all');
                    [~, temp_label] = max(temp_prob, [], 2);
                    Y_pred(i, j) = temp_label;
                end
            end
        end
        
        function Y_pred = predict(obj, X)
            % Purpose: To predict the behavior labels for a given data
            %
            % Input:
            % X: array with dimensions num_input_seq*num_input_dim*time
            %     the evidence sequences for various trials across time
            %
            % Output:
            % Y_pred: array with dimensions num_input_seq*time
            %     the predicted label configuration sequences for various
            %     trials across time
            
            % Code:
            state_seq = zeros(size(X, 1), size(X, 3));
            state_seq(:, 1) = obj.init_state_gen(size(X, 1));
            for i = 1: size(X, 1)
                sample = squeeze(X(i, :, :));
                transit_prob = obj.alpha(sample);
                for j = 2: size(X, 3)
                    [~, temp_val] = max(squeeze(transit_prob(state_seq(i, j-1), :, j)), [], 2);
                    state_seq(i, j) = temp_val;
                end
            end
            Y_pred = zeros(size(X, 1), size(X, 2));
            for i = 1: size(X, 1)
                for j = 1: size(X, 3)
                    sample = [squeeze(X(i, :, j)), 1];
                    temp_val = sample*squeeze(obj.F_emission(:, state_seq(i, j), :));
                    temp_prob = exp(temp_val)/sum(exp(temp_val), 'all');
                    [~, temp_label] = max(temp_prob, [], 2);
                    Y_pred(i, j) = temp_label;
                end
            end
        end
        
        function likelihood_out = predict_likelihood_onestep(obj, X, Y)
            % Purpose: To predict the likelihood of the output labels for a
            %          given evidence sequence using the one-step
            %          prediction algorithm
            %
            % Input:
            % X: array with dimensions num_input_dim*time
            %     depicting the evidence sequence for a particular trial
            % Y: array with dimensions 1*time 
            %     depicting the label configuration sequence for the given
            %     trial
            %
            % Output: 
            % likelihood_out: array with dimensions num_output_dim*time
            %     the likelihoods of each of the potential outputs across time
            
            % Code
            likelihood_out = zeros(obj.num_output, size(X, 2));
            hidden_state_seq = zeros(1, size(X, 2));
            for t = 1: size(X, 2)
                if t == 1
                    state_prob = obj.Pi;
                else
                    state_prob = exp(squeeze(obj.F_transition(:, hidden_state_seq(1, t-1), :))'*vertcat(X(:, t-1), 1));
                    state_prob = state_prob/sum(state_prob);
                end
                likelihood_local = zeros(obj.num_hidden, obj.num_output);
                for i = 1: obj.num_output
                    likelihood_local(:, i) = exp(squeeze(obj.F_emission(:, :, i))'*vertcat(X(:, t), 1));
                    likelihood_local(:, i) = likelihood_local(:, i)/sum(likelihood_local(:, i));
                end
                likelihood_out(:, t) = likelihood_local'*state_prob;
                [~, temp_val] = max(likelihood_local(:, Y(1, t)));
                hidden_state_seq(1, t) = temp_val;
            end
        end
        
        function likelihood_out = predict_likelihood_nohistory(obj, X)
            % Purpose: To predict the likelihood of the output labels for a
            %          given evidence sequence without using any part of
            %          the ground truth
            %
            % Input:
            % X: array with dimensions num_input_dim*time
            %     depicting the evidence sequence for a particular trial
            % Y: array with dimensions 1*time 
            %     depicting the label configuration sequence for the given
            %     trial
            %
            % Output: 
            % likelihood_out: array with dimensions num_output_dim*time
            %     the likelihoods of each of the potential outputs across time
            
            % Code
            likelihood_out = zeros(obj.num_output, size(X, 2));
            for t = 1: size(X, 2)
                if t == 1
                    state_prob = obj.Pi;
                else
                    state_prob = zeros(obj.num_hidden, obj.num_hidden);
                    for i = 1: obj.num_hidden
                        state_prob(:, i) = exp(squeeze(obj.F_transition(:, i, :))'*vertcat(X(:, t-1), 1));
                        state_prob(:, i) = state_prob(:, i)/sum(state_prob(:, i));
                    end
                    state_prob = sum(state_prob, 2);
                end
                likelihood_local = zeros(obj.num_hidden, obj.num_output);
                for i = 1: obj.num_output
                    likelihood_local(:, i) = exp(squeeze(obj.F_emission(:, :, i))'*vertcat(X(:, t), 1));
                    likelihood_local(:, i) = likelihood_local(:, i)/sum(likelihood_local(:, i));
                end
                likelihood_out(:, t) = likelihood_local'*state_prob;
            end
        end
        
        % Identifying Optimal State Sequence
        function state_seq = HSS(obj, X, Y)
            
            % Purpose: To estimate the most likely hidden state sequence
            %          for a given label sequence and corresponding
            %          evidence sequence
            %
            % Input:
            % X: array with dimensions num_input_dim*time
            %     depicting the evidence sequence for a particular trial
            % Y: array with dimensions 1*time 
            %     depicting the label configuration sequence for the given
            %     trial
            %
            % Output: 
            % state_seq: array with dimensions 1*time
            %     the most likely hidden state for each time point
            
            % Code
            alpha = obj.alpha(X);
            eta = obj.eta(X);
            [a_t, c_t] = obj.forward(X, Y, alpha, eta, 1, 1);
            b_t = obj.backward(X, Y, alpha, eta, 1, c_t);
            gamma = obj.gamma(a_t, b_t);
            [~, state_seq] = max(gamma, [], 1);
        end
        
        function state_seq = HSS_viterbi(obj, X, Y)
            
            % Purpose: To estimate the most likely hidden state sequence
            %          for a given label sequence and corresponding
            %          evidence sequence, using the viterbi algorithm
            %
            % Input:
            % X: array with dimensions num_input_dim*time
            %     depicting the evidence sequence for a particular trial
            % Y: array with dimensions 1*time 
            %     depicting the label configuration sequence for the given
            %     trial
            %
            % Output: 
            % state_seq: array with dimensions 1*time
            %     the most likely hidden state for each time point
            
            % Code
            state_seq = zeros(1, size(X, 2));
            alpha = obj.alpha(X);
            eta = obj.eta(X);
            delta = zeros(obj.num_hidden, size(X, 2));
            psi = zeros(obj.num_hidden, size(X, 2));
            
            % Initialization
            delta(:, 1) = obj.Pi.*eta(:, Y(1, 1), 1);
            psi(:, 1) = zeros(obj.num_hidden, 1);
            
            % Recursion
            for t = 2: size(X, 2)
                for k = 1: obj.num_hidden
                    [max_val, max_id] = max(delta(:, t-1).*squeeze(alpha(:, k, t-1))*eta(k, Y(1, t), t));
                    delta(:, t) = max_val;
                    psi(:, t) = max_id;
                end
            end
            
            % Termination
            [best_path_prob, best_path_final_state] = max(delta(:, end));
            
            state_seq(1, end) = best_path_final_state;
            for t = size(X, 2)-1: -1: 1
                state_seq(1, t) = psi(state_seq(1, t+1), t+1);
            end
        end
        
        % Transition & Emission Probability Functions
        function probability_array = alpha(obj, X)
            
            % Purpose: To compute the state transition probabilities for
            %          the given sequence
            %
            % Input:
            % X: array with dimensions num_input_dim*time
            %     depicting the evidence sequence for a particular trial
            %
            % Output:
            % probability_array: arrary with dimensions
            %                    num_hidden*num_hidden*time
            %     the transition probabilities
            
            % Code:
            T = size(X, 2);
            K = obj.num_hidden;
            probability_array = zeros(K, K, T);
            epsilon = 0.0001;
            for t = 1: T
                for c = 1: K
                    for c_dash = 1: K
                        probability_array(c, c_dash, t) = exp(obj.F_transition(:, c, c_dash)'*vertcat(X(:, t), 1)) + epsilon;
                    end
                    probability_array(c, :, t) = probability_array(c, :, t)/sum(probability_array(c, :, t));
                end
            end
        end
        
        function probability_array = eta(obj, X)
            
            % Purpose: To compute the emission probabilities for the given
            %          sequence for all possible states
            %
            % Input:
            % X: array with dimensions num_input_dim*time
            %     depicting the evidence sequence for a particular trial
            %
            % Output: 
            % probability_array: array with dimensions
            %                    num_hidden*num_output*time
            %     the emission probabilities
            
            % Code:
            epsilon = 0.01;
            K = obj.num_hidden;
            J = obj.num_output;
            T = size(X, 2);
            probability_array = zeros(K, J, T);
            for t = 1: T
                for c = 1: K
                    for j = 1: J
                        if ~isinf(exp(obj.F_emission(:, c, j)'*vertcat(X(:, t), 1)) + epsilon)
                            probability_array(c, j, t) = exp(obj.F_emission(:, c, j)'*vertcat(X(:, t), 1)) + epsilon;
                        else
                            probability_array(c, j, t) = exp(709);
                        end
                    end
                    probability_array(c, :, t) = probability_array(c, :, t)/sum(probability_array(c, :, t));
                end
            end
        end
        
        
        % Helper functions
        
        % Forward algorithm
        function [step_prob_array, scales_sequence] = forward(obj, X, Y, alpha, eta, varargin)
            
            % Purpose: To compute the forward marginal probabilities for
            %          the given sequence
            %
            % Input:
            % X: array with dimensions num_input_dim*time
            %     depicting the evidence sequence for a particular trial
            % Y: array with dimensions 1*time 
            %     depicting the label configuration sequence for the given
            %     trial
            % alpha: array with dimensions num_hidden*num_hidden*time
            %     the transition probabilities
            % eta: array with dimensions num_hidden*num_output*time
            %     the emission probabilities
            %
            % Optional Input:
            % scaled: boolean
            %     whether or not to use the scaled version of the forward
            %     algorithm; TIP: set this to true if the output of the
            %     function has rapidly decaying values in time, which would
            %     usually happen if T > 10
            %
            % Output:
            % step_prob_array: array with dimensions num_hidden*time
            %     the forward-step marginal probabilities for the given
            %     sequence
            % scales_sequence: array with dimensions 1*time
            %     the scaling constants
            
            % Code:
            
            % taking in optional input
            nVarargs = length(varargin);
            if nVarargs > 0
                scaled = varargin{1};
            else
                scaled = 0;
            end
            
            K = obj.num_hidden;
            T = size(X, 2);
            Pi = obj.Pi;
            step_prob_array = zeros(K, T);
            for c = 1: K
                step_prob_array(c, 1) = Pi(c, 1)*eta(c, Y(1, 1)+1, 1);
            end
            if scaled
                scales_sequence = zeros(1, T);
                scales_sequence(1, 1) = 1/sum(step_prob_array(:, 1), 'all');
                step_prob_array(:, 1) = step_prob_array(:, 1)*scales_sequence(1, 1);
            end
            if ~scaled
                for t = 2: T
                    for c = 1: K
                        step_prob_array(c, t) = (step_prob_array(:, t-1)'*alpha(:, c, t-1))*eta(c, Y(1, t)+1, t); %debug
                        if step_prob_array(c, t) == 0
                            step_prob_array(c, t) = 10^-323;
                        end
                    end
                end
            elseif scaled
                for t = 2: T
                    for c = 1: K
                        step_prob_array(c, t) = (step_prob_array(:, t-1)'*alpha(:, c, t-1))*eta(c, Y(1, t)+1, t); %debug
                        if step_prob_array(c, t) == 0
                            step_prob_array(c, t) = 10^-323;
                        end
                    end
                    scales_sequence(1, t) = 1/sum(step_prob_array(:, t), 'all');
                    if scales_sequence(1, t) == Inf
                        scales_sequence(1, t) = 10^308;
                    end
                    step_prob_array(:, t) = step_prob_array(:, t)*scales_sequence(1, t);
                end
            end
        end
        
        % Backward algorithm
        function step_prob_array = backward(obj, X, Y, alpha, eta, varargin)
            
            % Purpose: To compute the forward marginal probabilities for
            %          the given sequence
            %
            % Input:
            % X: array with dimensions num_input_dim*time
            %     depicting the evidence sequence for a particular trial
            % Y: array with dimensions 1*time 
            %     depicting the label configuration sequence for the given
            %     trial
            % alpha: array with dimensions num_hidden*num_hidden*time
            %     the transition probabilities
            % eta: array with dimensions num_hidden*num_output*time
            %     the emission probabilities
            %
            % Optional Input:
            % scaled: boolean
            %     whether or not to use the scaled version of the forward
            %     algorithm; TIP: set this to true if the output of the
            %     function has rapidly decaying values in time, which would
            %     usually happen if T > 10
            % return_scales: boolean 
            %     whether or not to return the sequence of scaling
            %     constants used
            %
            % Output: 
            % step_prob_array: array with dimensions num_hidden*time
            %     the backward-step marginal probabilities for the given
            %     sequence
            
            % Code:
            
            % taking in optional inputs
            nVarargs = length(varargin);
            if nVarargs > 0
                scaled = varargin{1};
                scales_sequence = varargin{2};
            end
            
            K = obj.num_hidden;
            T = size(X, 2);
            step_prob_array = zeros(K, T);
            step_prob_array(:, T) = ones(K, 1);
            for t_opp = 1: T-1
                t = T - t_opp;
                for c = 1: K
                    step_prob_array(c, t) = (squeeze(alpha(c, :, t))'.*eta(:, Y(1, t+1)+1, t+1))'*step_prob_array(:, t+1);
                    if step_prob_array(c, t) == 0
                        step_prob_array(c, t) = 10^-323;
                    end
                end
            end
            if scaled
                for t = 1: T
                    step_prob_array(:, t) = step_prob_array(:, t)*scales_sequence(1, t);
                end
            end
        end
        
        function marginal_prob_out = gamma(obj, a_t, b_t)
            
            % Purpose: To compute the single marginal probabilities of the
            %          posterior distribution of the state sequence
            %
            % Input: 
            % a_t: array with dimensions num_hidden*time array
            %     the forward probabilities of the sequence of interest
            % b_t: array with dimensions num_hidden*time
            %     the backward probabilities of the sequence of interest
            %
            % Output:
            % marginal_prob_out: array with dimensions num_hidden*time
            %     the probabilities of the sequence of interest
            
            % Code:
            epsilon = 10^(-323);
            K = obj.num_hidden; 
            T = size(a_t, 2);
            marginal_prob_out = zeros(K, T);
            for t = 1: T
                marginal_prob_out(:, t) = a_t(:, t).*b_t(:, t) + epsilon;
                marginal_prob_out(:, t) = marginal_prob_out(:, t)/sum(marginal_prob_out(:, t));
            end
        end
        
        function marginal_prob_out = zeta(obj, Y, a_t, b_t, alpha, eta)
        
            % Purpose: To compute the pairwise marginal probabilities of
            %          the posterior distribution of the state sequence
            %
            % Input:
            % Y: array with dimensions 1*time 
            %     depicting the label configuration sequence for the given
            %     trial
            % a_t: array with dimensions num_hidden*time array
            %     the forward probabilities of the sequence of interest
            % b_t: array with dimensions num_hidden*time
            %     the backward probabilities of the sequence of interest
            % alpha: array with dimensions num_hidden*num_hidden*time
            %     the transition probabilities
            % eta: array with dimensions num_hidden*num_output*time
            %     the emission probabilities
            %
            % Output:
            % marginal_prob_out: array with dimensions num_hidden*time
            %     the probabilities of the sequence of interest
            
            % Code:
            epsilon = 10^(-8);
            K = obj.num_hidden;
            T = size(a_t, 2);
            marginal_prob_out = zeros(K, K, T);
            for t = 1: T-1
                for c = 1: K
                    for c_dash = 1: K
                        marginal_prob_out(c, c_dash, t) = a_t(c, t)*b_t(c_dash, t+1)*alpha(c, c_dash, t+1)*eta(c_dash, Y(1, t+1)+1, t+1);
                    end
                end
                marginal_prob_out(:, :, t) = marginal_prob_out(:, :, t)/(sum(marginal_prob_out(:, :, t), 'all') + epsilon);
            end
        end
        
        % Others
        function states = init_state_gen(obj, n)
            
            % Purpose: To generate uniformly distributed samples for
            %          initial states based on Pi
            %
            % Input:
            % n: positive integer
            %     number of states to generate
            %
            % Output:
            % states: array with dimensions 1*n
            %     array with the intialized values for initial state
            %     probabilities
            
            % Code:
            states = [];
            for i = 1: n
                temp_val = rand;
                if temp_val < obj.Pi(1, 1)
                    states = [states, 1];
                else
                    for j = 2: obj.num_hidden
                        if all([temp_val>=sum(obj.Pi(1:j-1, 1), 'all'), temp_val<sum(obj.Pi(1:j), 'all')])
                            states = [states, j];
                            break;
                        end
                    end
                end
            end
        end
        
    end
    
end