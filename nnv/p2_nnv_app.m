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


%testranges = [\
%(0.000087 ,0.000115),
%(0.027858 ,0.036758),
%(0.111430 ,0.147033),
%(0.019698 ,0.051984),
%(0.078793 ,0.207937),
%(0.630346 ,1.663493),
%(-0.000115,-0.000087),
%(-0.036758,-0.027858),
%(-0.147033,-0.111430),
%(-0.051984,-0.019698),
%(-0.207937,-0.078793),
%(-1.663493,-0.630346)]


%%
range_select_idx = 4;
ilb = 0.078793;%0.11143;%0.019698;%0.027858;%0.000087;%testranges(range_select_idx, 1);
iub = 0.207937;%0.147033;%0.051984;%0.036758;%0.000115;%testranges(range_select_idx, 2);

pulse_idx = 1;
pulse_ilb_abs = 0.5;
pulse_iub_abs = 1.0;

if ilb > 0
  pulse_ilb = -pulse_iub_abs;
  pulse_iub = -pulse_ilb_abs;
else
  pulse_ilb = pulse_ilb_abs;
  pulse_iub = pulse_iub_abs;
end

lb = [init_state repmat(ilb,1,pulse_idx-1) pulse_ilb repmat(ilb,1,N_stimulus-pulse_idx)]';
ub = [init_state repmat(ilb,1,pulse_idx-1) pulse_iub repmat(iub,1,N_stimulus-pulse_idx)]';
lb = double(lb); % changle to double
ub = double(ub); % changle to double
initial_star = Star(lb, ub);

Threshold = 100;

prev_stars = initial_star;
for idx = 1:N_stimulus
    layer = constructStimulusLayer(W_rec, b_rec, W_in, N_stimulus-idx+1);
    one_layer = FFNNS(layer);
    prev_stars = one_layer.reach(prev_stars);
    s = size(prev_stars);
    s(1,2)
    if s(1,2)  > Threshold
      X = Star.get_hypercube_hull(prev_stars);
      prev_stars = X.toStar;
    end
end

for idx = 1:N_settle
    settle_layer = constructSettleLayer(W_rec, b_rec, W_in);
    one_settle_layer = FFNNS(settle_layer);
    prev_stars = one_settle_layer.reach(prev_stars);
    s = size(prev_stars);
    if s(1,2) > Threshold
      X = Star.get_hypercube_hull(prev_stars);
      prev_stars = X.toStar;
    end
end

output_layer = constructOutputLayer(W_out, b_out);
one_output_layer = FFNNS(output_layer);

final_output = one_output_layer.reach(prev_stars);
%final_box = final_output.getBox;
%display(final_box.lb)
%display(final_box.ub)
n_final_poly = size(final_output);
display(n_final_poly)
%display(final_output.getBox)
n_final_poly = n_final_poly(1,2);
display(n_final_poly)
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

display (n_pos)
display (n_neg)
display (n_unknown)

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
  L = LayerS(W_nn, B_nn, 'poslin');
end

function L = constructSettleLayer(W_rec, b_rec, W_in)
  settle_I = 1;
  W_in_select = 2;
  W_i = W_in(W_in_select,:)';
  B_nn = b_rec' + W_i * settle_I; 
  W_nn = W_rec';
  L = LayerS(W_nn, B_nn, 'poslin');  
end

function L = constructOutputLayer(W_out, b_out)
  L = LayerS(W_out', b_out, 'purelin');
end

