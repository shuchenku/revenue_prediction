%% decision tree

% 
% Indices = crossvalind('Kfold', length(nonoutliers), 10);
% full = 1:length(nonoutliers);
% 
% new_tmp_all = full_train(nonoutliers,:);
% new_Revenue = Revenue(nonoutliers,:);
% 
% precision = zeros(19,10);
% 
% for i = 1:10
%     test_idx = find(Indices==i);
%     train_idx = setdiff(full,test_idx);
%     
%     test = new_tmp_all(test_idx,:);
%     train = new_tmp_all(train_idx,:);
%     testrev = new_Revenue(test_idx,:);
%     trainrev = new_Revenue(train_idx,:);
%     
%     tree = fitrtree(train,trainrev,'CategoricalPredictors',[1,2,3,cate]);
%     
%     for lvl = 1:max(tree.PruneList)
%         treep = prune(tree,'level',lvl);
%         output = predict(treep,test);
%         error = mse(output-testrev);
%         precision(lvl,i) = error;
%     end
% end

%% boosting/bagging
% 
Indices = crossvalind('Kfold', length(nonoutliers), 3);
full = 1:length(nonoutliers);


new_tmp_all = full_train(nonoutliers,2:end);
new_Revenue = Revenue(nonoutliers);

% 
% Indices = crossvalind('Kfold', length(nonoutliers)-1, 5);
% full = 1:length(nonoutliers)-1;
% 
% 
% new_tmp_all = full_train([nonoutliers,125],:);
% new_Revenue = Revenue([nonoutliers,125],:);



std_mat = zeros(1,5);
num_of_trees = [100,200,300,400,500];
% num_of_trees = [50,100,200,300,400,500,600,700,800];
% learn_rates = [0.01,0.1,0.2];
% learn_rates = [0.01,0.1,0.2,0.4,0.8,1];
learn_rates = 0.1;


for  lr = 1:length(learn_rates)
    
    for nt = 1:length(num_of_trees)
        
        error = zeros(1,3);
        
        for i = 1:3
            test_idx = find(Indices==i);
            train_idx = setdiff(full,test_idx);
            test = new_tmp_all(test_idx,:);
            train = new_tmp_all(train_idx,:);
            testrev = new_Revenue(test_idx,:);
            trainrev = new_Revenue(train_idx,:);
%             predictor = fitensemble(train, trainrev, 'LSBoost',num_of_trees(nt), 'Tree', 'Type', 'Regression','CategoricalPredictors',[1,2,3],'LearnRate',learn_rates(lr));
            predictor = fitensemble(train, trainrev, 'LSBoost',num_of_trees(nt), 'Tree', 'Type', 'Regression','CategoricalPredictors',2,'Resample','on','LearnRate',0.1);
%              predictor = fitensemble(train, trainrev, 'Bag',num_of_trees(nt), 'Tree', 'Type', 'Regression','CategoricalPredictors',1);
            output = predict(predictor,test);
                    display(i)
            error(i) = mse(output-testrev);
        end
        
        std_mat(lr,nt) = mean(error);
        
    end
end


% predictor = fitensemble(full_train(nonoutliers,:), Revenue(nonoutliers), 'Bag', 500, 'Tree', 'Type', 'Regression','CategoricalPredictors',[1,2,3,cate,42]);

%% std classify
% 
% Indices = crossvalind('Kfold', length(std_class), 3);
% full = 1:length(std_class);
% 
% 
% new_tmp_all = full_train;
% new_Revenue = std_class;

% 
% Indices = crossvalind('Kfold', length(nonoutliers)-1, 5);
% full = 1:length(nonoutliers)-1;
% 
% 
% new_tmp_all = full_train([nonoutliers,125],:);
% new_Revenue = Revenue([nonoutliers,125],:);
% 
% 
% 
% std_mat = zeros(3,7);
% num_of_trees = [50,100,150,200,300,400,500];
% % num_of_trees = [50,100,200,300,400,500,600,700,800];
% % learn_rates = [0.01,0.1,0.15];
% % learn_rates = [0.01,0.1,0.2,0.4,0.8,1];
% learn_rates = 1;
% 
% for  lr = 1:length(learn_rates)
%     
%     for nt = 1:length(num_of_trees)
%         
%         error = zeros(1,3);
%         
%         for i = 1:3
%             test_idx = find(Indices==i);
%             train_idx = setdiff(full,test_idx);
%             test = new_tmp_all(test_idx,:);
%             train = new_tmp_all(train_idx,:);
%             testrev = new_Revenue(test_idx,:);
%             trainrev = new_Revenue(train_idx,:);
%             predictor = fitensemble(train, trainrev,'LPBoost',num_of_trees(nt), 'Tree','CategoricalPredictors',[1,2,3,cate]);
% %              predictor = fitensemble(train, trainrev, 'Bag',num_of_trees(nt), 'Tree', 'Type', 'Regression','CategoricalPredictors',[1,2,3,cate,42]);
%             output = predict(predictor,test);
%             tmp = output-testrev;
%             error(i) = length(find(tmp==0))/137;
%         end
%         
%         std_mat(lr,nt) = mean(error);
%         
%     end
% end