param.s = 3; 					% epsilon SVR
param.C = max(Revenue) - min(Revenue);	% FIX C based on Equation 9.61
param.t = 2; 					% RBF kernel
param.gset = 2.^[-7:7];				% range of the gamma parameter
param.eset = [0:5];				% range of the epsilon parameter
param.nfold = 5;				% 5-fold CV

Rval = zeros(length(param.gset), length(param.eset));

data = [Revenue,full_train(:,2:end)];
Indices = crossvalind('Kfold', length(nonoutliers), 3);
full = 1:length(nonoutliers);


for i = 1:param.nfold
    % partition the training data into the learning/validation
    % in this example, the 5-fold data partitioning is done by the following strategy,
    % for partition 1: Use samples 1, 6, 11, ... as validation samples and
    %			the remaining as learning samples
    % for partition 2: Use samples 2, 7, 12, ... as validation samples and
    %			the remaining as learning samples
    %   :
    % for partition 5: Use samples 5, 10, 15, ... as validation samples and
    %			the remaining as learning samples
    
    test_idx = find(Indices==i);
    train_idx = setdiff(full,test_idx);
    valdata.X = new_tmp_all(test_idx,:);
    lrndata.X = new_tmp_all(train_idx,:);
    valdata.y = new_Revenue(test_idx,:);
    lrndata.y = new_Revenue(train_idx,:);
    
    for j = 1:length(param.gset)
        param.g = param.gset(j);
        
        for k = 1:length(param.eset)
            param.e = param.eset(k);
            param.libsvm = ['-s ', num2str(param.s), ' -t ', num2str(param.t), ...
                ' -c ', num2str(param.C), ' -g ', num2str(param.g), ...
                ' -p ', num2str(param.e)];
            
            % build model on Learning data
            model = svmtrain(lrndata.y, lrndata.X, param.libsvm);
            
            % predict on the validation data
            [y_hat, Acc, projection] = svmpredict(valdata.y, valdata.X, model);
            
            Rval(j,k) = Rval(j,k) + mean((y_hat-valdata.y).^2);
        end
    end
    
end

Rval = Rval ./ (param.nfold);