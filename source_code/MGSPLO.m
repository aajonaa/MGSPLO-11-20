function [Best_pos, Convergence_curve] = MGSPLO(N, MaxFEs, lb, ub, dim, fobj)
% Multi-dimensional Gradient interaction Search PLO (MGSPLO)
% dynamic gradient interaction (AGDO)
% Multi-Dimensional Search Pattern, Preying Behavior (SFOA)
%% Initialization
FEs = 0;
it = 1;
fitness = inf * ones(N, 1);
fitness_new = inf * ones(N, 1);

% Ensure lb and ub are row vectors
if size(lb, 1) > 1; lb = lb'; end
if size(ub, 1) > 1; ub = ub'; end

X = initialization(N, dim, ub, lb);
V = ones(N, dim); % From original PLO
X_new = zeros(N, dim);

% --- DGIS (Adam) Parameter Initialization ---
% These are required for the new DGIS strategy (Strategy 1)
lr = 0.01;      % Adam learning rate
beta1 = 0.9;    % Adam first moment decay
beta2 = 0.999;  % Adam second moment decay
epsilon = 1e-8; % Adam epsilon
m = zeros(N, dim); % Adam first moment vector (one per agent)
v = zeros(N, dim); % Adam second moment vector (one per agent)
% --------------------------------------------

for i = 1:N
    fitness(i) = fobj(X(i, :));
    FEs = FEs + 1;
end

[fitness, SortOrder] = sort(fitness);
X = X(SortOrder, :);

Bestpos = X(1, :);
Bestscore = fitness(1);
Convergence_curve = [];
Convergence_curve(it) = Bestscore;

%% Main loop
while FEs < MaxFEs
    
    % --- Calculate PLO Backbone Parameters ---
    X_sum = sum(X, 1);
    X_mean = X_sum / N;
    
    % Dynamic weights from original PLO
    t_ratio = FEs / MaxFEs; % Current time ratio
    w1 = tansig(t_ratio^4);
    w2 = exp(-(2 * t_ratio)^3);
    
    % Probability of exploitation increases as FEs increase
    p_exploit = t_ratio^2;
    
    % Module 1 (High-D Explore) parameter: Angle narrows over time
    theta = pi * (1 - t_ratio);
    
    % Module 2 (Low-D Explore) parameter: Energy decreases over time
    tEO = 1 - t_ratio;

    df = randperm(N, 5);
    dm = zeros(5, dim); % Pre-allocate
    dm(1, :) = Bestpos - X(df(1), :);
    dm(2, :) = Bestpos - X(df(2), :);
    dm(3, :) = Bestpos - X(df(3), :);
    dm(4, :) = Bestpos - X(df(4), :);
    dm(5, :) = Bestpos - X(df(5), :);
    

    for i = 1:N
        if rand < p_exploit
            if rand > 0.8
                % --- Strategy 1: DGIS (Exploitation) ---
                % This strategy replaces the "Preying Behavior" from SFOAPLO.
                
                % Calculate 'a' perturbation vector - Eq. (17)
                a = (1 - t_ratio) * rand(1, dim);
                
                % Select two random indices (a1, a2) - Eq. (14)
                A1 = randperm(N);
                A1(A1 == i) = []; % Ensure a1 and a2 are not 'i'
                a1 = A1(1);
                a2 = A1(2);
                
                % Calculate the system direction pointer 'zeta' - Eq. (15)
                zeta = sign(fitness(a1) - fitness(i));
                if zeta == 0 % Handle case where fitness is equal
                    zeta = 1;
                end
                
                % Calculate 'P' (gradient proxy)
                % Using (X_mean - current_pos) as the gradient proxy
                P = X_mean - X(i, :);
                
                % Calculate 'f', the Adam-inspired "key reference point" G - Eq. (19)
                % This call updates m and v for agent 'i'
                [f, m(i, :), v(i, :)] = gtdt(Bestpos, P, lr, it, beta1, beta2, epsilon, m(i, :), v(i, :));
                
                % DGIS Case 1: Pre-exploitation update rule - Eq. (16)
                % (Using X(i,:) for 'npo_line' from the pseudocode)
                npo_1a = X(i, :) + zeta * a .* (f - X(a1, :)) - a .* (X(i, :) - X(a2, :));
                
                % DGIS Case 2: Post-exploitation update rule - Eq. (18)
                npo_1b = X(a1, :) + a .* (f - X(a2, :));
                
                % Stochastic switch between DGIS Case 1 and Case 2 - Eq. (24)
                % (Replacing 'rand/k > rand' with a simple 50/50 stochastic switch)
                X_new_i = zeros(1, dim);
                for j = 1:dim
                    if rand < 0.5
                        X_new_i(j) = npo_1b(j); % Use post-exploitation rule
                    else
                        X_new_i(j) = npo_1a(j); % Use pre-exploitation rule
                    end
                end
                X_new(i, :) = X_new_i;
            else
                % --- Strategy 1: Preying Behavior (Module 3 - Exploitation) ---
                r1 = rand;
                r2 = rand;
                % Select two random distances from the 5 calculated
                kp = randperm(5, 2);
                % Update position based on Eq. (11)
                X_new(i, :) = X(i, :) + r1 * dm(kp(1), :) + r2 * dm(kp(2), :);
            end
            
        else
            % Exploration Phase (Split 50/50 between PLO and Starfish)
            
            if rand < 0.5
                % --- Strategy 2a: Original PLO Backbone Update ---
                % This is the "backbone code" that must be preserved.
                a = rand() / 2 + 1;
                V(i, :) = 1 * exp((1 - a) / 100 * FEs); % Update V matrix row
                LS = V(i, :);
                GS = Levy(dim) .* (X_mean - X(i, :) + (lb + rand(1, dim) .* (ub - lb)) / 2);
                X_new(i, :) = X(i, :) + (w1 * LS + w2 * GS) .* rand(1, dim);
            
            else
                % --- Strategy 2b: Starfish Exploration (Module 1 or 2) ---
                temp_X = X(i, :); % Start with current position
                
                if dim > 5
                    % --- Module 1: Five-Dimensional Search Pattern (High-D) ---
                    jp1 = randperm(dim, 5); % Select 5 random dimensions
                    for j = 1:5
                        pm = (2 * rand - 1) * pi;
                        if rand < 0.5
                            temp_val = X(i, jp1(j)) + pm * (Bestpos(jp1(j)) - X(i, jp1(j))) * cos(theta);
                        else
                            temp_val = X(i, jp1(j)) - pm * (Bestpos(jp1(j)) - X(i, jp1(j))) * sin(theta);
                        end
                        % Boundary check: "stays in the previous position"
                        if temp_val <= ub && temp_val >= lb
                            temp_X(jp1(j)) = temp_val;
                        end
                    end
                else
                    % --- Module 2: Unidimensional Search Pattern (Low-D) ---
                    jp2 = ceil(dim * rand); % Select 1 random dimension
                    im = randperm(N, 2); % Get random indices for k1, k2
                    rand1 = 2 * rand - 1;
                    rand2 = 2 * rand - 1;
                    
                    temp_val = tEO * X(i, jp2) + ...
                                 rand1 * (X(im(1), jp2) - X(i, jp2)) + ...
                                 rand2 * (X(im(2), jp2) - X(i, jp2));
                    
                    % Boundary check: "stays in the previous position"
                    if temp_val <= ub && temp_val >= lb
                        temp_X(jp2) = temp_val;
                    end
                end
                X_new(i, :) = temp_X; % Assign the updated vector
            end
        end
    end
    % === End of Primary Update ===
    
    % =================================================================
    % === PLO Backbone: Mutation, Boundary, Evaluation, Selection ===
    % This entire block is from the original PLO and is preserved
    % to maintain the "backbone" structure.
    % =================================================================
    E = sqrt(t_ratio); % E = sqrt(FEs / MaxFEs)
    A = randperm(N);
    
    for i = 1:N
        % Secondary mutation (from original PLO)
        for j = 1:dim
            if (rand < 0.05) && (rand < E)
                X_new(i, j) = X_new(i, j) + sin(rand * pi) * (X_new(i, j) - X_new(A(i), j));
            end
        end
        
        % Standard Boundary Check (Clamping)
        Flag4ub = X_new(i, :) > ub;
        Flag4lb = X_new(i, :) < lb;
        X_new(i, :) = (X_new(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        
        % Evaluation
        if FEs >= MaxFEs; break; end % Check FEs budget
        fitness_new(i) = fobj(X_new(i, :));
        FEs = FEs + 1;
        
        % Greedy Selection
        if fitness_new(i) < fitness(i)
            X(i, :) = X_new(i, :);
            fitness(i) = fitness_new(i);
        end
    end
    % === End of PLO Backbone Block ===
    
    % Sort population and update global best
    [fitness, SortOrder] = sort(fitness);
    X = X(SortOrder, :);
    
    if fitness(1) < Bestscore
        Bestpos = X(1, :);
        Bestscore = fitness(1);
    end
    
    it = it + 1;
    Convergence_curve(it) = Bestscore;
    Best_pos = Bestpos;
    
    if FEs >= MaxFEs; break; end % Final check for outer loop
end
end

%==================================================================
% --- Helper Functions (Required by PLO/IPLO_DGIS) ---
%==================================================================

function [f, m_new, v_new] = gtdt(bestpos, g, lr, it, beta1, beta2, epsilon, m_old, v_old)
% gtdt: Adam-inspired update function for DGIS
% Calculates a "key reference point" f based on an Adam-style update.
%
% Inputs:
%   bestpos - The current global best position
%   g       - The gradient proxy (e.g., X_mean - X(i,:))
%   lr      - Learning rate
%   it      - Current iteration (t) for bias correction
%   beta1   - Adam parameter
%   beta2   - Adam parameter
%   epsilon - Adam parameter
%   m_old   - Previous first moment vector
%   v_old   - Previous second moment vector
%
% Outputs:
%   f       - The new key reference point (a position vector)
%   m_new   - Updated first moment vector
%   v_new   - Updated second moment vector

% Update biased first and second moment estimates
m_new = beta1 * m_old + (1 - beta1) * g;
v_new = beta2 * v_old + (1 - beta2) * (g .^ 2);

% Compute bias-corrected estimates
% (Add small value to 'it' in case it=0, though it starts at 1)
m_hat = m_new / (1 - beta1^it + epsilon);
v_hat = v_new / (1 - beta2^it + epsilon);

% Calculate the update step
update_step = lr * m_hat ./ (sqrt(v_hat) + epsilon);

% 'f' is the new reference point based on the best position
% We use '-' as in standard gradient descent (moving "downhill")
f = bestpos - update_step;
end

%==================================================================

function o = Levy(d)
% Levy flight generator (from original PLO)
beta = 1.5;
sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
u = randn(1, d) * sigma;
v = randn(1, d);
step = u ./ abs(v).^(1 / beta);
o = step;
end

%==================================================================

function X = initialization(N, dim, ub, lb)
% Standard initialization function (from original PLO)
X = zeros(N, dim);
for i = 1:N
    X(i, :) = lb + (ub - lb) .* rand(1, dim);
end
end