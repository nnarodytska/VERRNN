% This script performs rechability analysis
% Hongce Zhang

tic
%%
load rnn
% gives W_rec W_out b_rec b_out W_in init_state
load ranges
% testranges

N_stimulus = 50;
N_settle = 50;

W_in = double(W_in); % change to double
W_rec = double(W_rec); % change to double

%%
range_select_idx = 3;
ilb = 0.315173;%testranges(range_select_idx, 1);
iub = 0.8311746;%testranges(range_select_idx, 2);

lb = [init_state repmat(ilb,1,N_stimulus)]';
ub = [init_state repmat(iub,1,N_stimulus)]';
lb = double(lb); % changle to double
ub = double(ub); % changle to double
initial_star = Star(lb, ub);
initial_star = initial_star.toPolyhedron;

layers = []; %zeros(N_stimulus+N_settle+1,1);
for idx = 1:N_stimulus
    layers = [layers constructStimulusLayer(W_rec, b_rec, W_in, N_stimulus-idx+1)];
end

for idx = 1:N_settle
    layers = [layers constructSettleLayer(W_rec, b_rec, W_in)];
end

layers = [layers constructOutputLayer(W_out, b_out)];
rl_controller = FFNN(layers);

sts = rl_controller.reach(initial_star, 'exact', 1);
final_output = sts;
n_final_poly = size(final_output);
n_final_poly = n_final_poly(1,2);
n_pos = 0;
n_neg = 0;
n_unknown = 0;
for idx = 1:n_final_poly
    final_box = final_output(1,idx).getBox;
    display(final_box.lb)
    display(final_box.ub)
    if final_box.lb * final_box.ub < 0
      display(final_box.lb)
      display(final_box.ub)
      n_unknown = n_unknown + 1;
    else
        if final_box.lb > 0
            n_pos = n_pos + 1;
        end
        if final_box.ub < 0
            n_neg = n_neg + 1;
        end
    end
end

display(n_unknown);
display (n_pos)
display (n_neg)
toc

function L = constructStimulusLayer(W_rec, b_rec, W_in, extra_input)
% input extra_input % W_in_select == 1 or 2
  num_state = size(W_rec,1);
  W_in_select = 1;
  preserve_output = extra_input-1; % extra_input finally should be 1
  W_i = W_in(W_in_select,:)';
  z1 = zeros(num_state, preserve_output);
  z2 = zeros(preserve_output,num_state+1);
  o  = eye(preserve_output);
  W_nn=[W_rec' W_i z1;z2 o];
  
  B_nn = zeros(num_state+preserve_output,1);
  B_nn(1:num_state) = b_rec'; 
  L = Layer(W_nn, B_nn, 'ReLU');
end

function L = constructSettleLayer(W_rec, b_rec, W_in)
  settle_I = 1;
  W_in_select = 2;
  W_i = W_in(W_in_select,:)';
  B_nn = b_rec' + W_i * settle_I; 
  W_nn = W_rec';
  L = Layer(W_nn, B_nn, 'ReLU');  
end

function L = constructOutputLayer(W_out, b_out)
  L = Layer(W_out', b_out, 'Linear');
end

