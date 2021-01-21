%% Analyze temp at t0 in relate to the characteristics

savepath = 'C:\Users\ido\Google Drive\Thesis\Data\Results\Charectarisitcs analysis\Vs BaseLine\';
filesPath='C:\Projects\Thesis\Feature-Analysis- matlab\JPGfeatureDataFile.mat';
load(filesPath);
type= 'BaseLine';% Or 'grav2Base'
chaPat='C:\Users\ido\Google Drive\Thesis\Data\Characteristics.xlsx';
characteristicTable = readtable(chaPat);

analyzedCharectTitle = {'Gender_Annova', 'Age', 'High', 'Weight','Is_smoking_Annova',...
    'Phys_Act_Freq_Annova','Phys_Act_Intence_Annova', 'Pulse_Presure'};
%{'Gender', 'Age', 'High', 'weight', 'Is_smoking',...
%'Phys_Act_Freq','Phys_Act_Intence', 'Pulse_Presure'};
nameOfCharactaristic ={'Gender_m_0_F_1_', 'Age', 'High_cm_','Weight_kg_', 'Is_smoking_', ...
   'PhysicalActivity_frequency_0_0_1_2_1_2_3_2_3AndUp_3', 'Physical_activity_intensit_light_0_Mild_1Intens_2_','PP_SBP_DBP_'};
currentAnalysis = 'intence';
allCurrentAnaInds = cellfun(@(x) ~isempty(x), cellfun(@(x) strfind(lower(x), currentAnalysis), variableNames,'UniformOutput',false));

allROIs = variableNames(allCurrentAnaInds);
leftMat(1:size(characteristicTable,1), 1:2, 1:size(allROIs,2)) = nan;
rightMat=leftMat;%[rows = patient #
% cols = 1- char Val,   2- data from image
% Depth - ROI]

%% Collect the data
for charInd = 1:length(analyzedCharectTitle)
    curCharTitle = analyzedCharectTitle{charInd};
    curChageName = nameOfCharactaristic{charInd};
    if strcmp(type, 'grav2Base')
        for paInd = 1:size(characteristicTable,1)
            currentId = characteristicTable.Ext_name{paInd};
            leftMat(paInd,1,:) = characteristicTable.(curChageName)(paInd); % Change struct field according to the analyzedCarect
            rightMat(paInd,1,:)= characteristicTable.(curChageName)(paInd);% Change struct field according to the analyzedCarect
            matchInd = find(cellfun(@(x) strcmp(x, currentId), correspondedId));
            
            if ~isempty(matchInd)
                for ind =matchInd'
                    currentHand = correspondedHandSide{ind};
                    
                    if strcmp(currentHand,'left')
                        
                        for ROIind = 1:size(allROIs,2)
                            
                            dataId = find(cellfun(@(x) x==1, cellfun(@(x) strcmpi(lower(x), allROIs{ROIind}), variableNames,'UniformOutput',false)));
                            BaseLine = allFeatureMat(1,dataId,ind);
                            endGrav = allFeatureMat(find(allFeatureMat(:,1,ind)==2,1,'first'), dataId,ind);
                            leftMat(paInd, 2, ROIind) = endGrav-BaseLine;
                            
                        end
                        
                    elseif strcmp(currentHand,'right')
                        for ROIind = 1:size(allROIs,2)
                            
                            dataId = find(cellfun(@(x) x==1, cellfun(@(x) strcmpi(lower(x), allROIs{ROIind}), variableNames,'UniformOutput',false)));
                            BaseLine = allFeatureMat(1,dataId,ind);
                            endGrav = allFeatureMat(find(allFeatureMat(:,1,ind)==2,1,'first'), dataId,ind);
                            rightMat(paInd, 2, ROIind) = allFeatureMat(1,dataId,ind);
                            
                        end
                        
                    else
                        error('Error with hand side');
                    end
                    
                    
                end
                
            end
            
        end
        % Compare with BaseLine
    elseif strcmp(type,'BaseLine')
        for paInd = 1:size(characteristicTable,1)
            currentId = characteristicTable.Ext_name{paInd};
            leftMat(paInd,1,:) = characteristicTable.(curChageName)(paInd); % Change struct field according to the analyzedCarect
            rightMat(paInd,1,:)= characteristicTable.(curChageName)(paInd);% Change struct field according to the analyzedCarect
            matchInd = find(cellfun(@(x) strcmp(x, currentId), correspondedId));
            
            if ~isempty(matchInd)
                for ind =matchInd'
                    currentHand = correspondedHandSide{ind};
                    
                    if strcmp(currentHand,'left')
                        
                        for ROIind = 1:size(allROIs,2)
                            
                            dataId = find(cellfun(@(x) x==1, cellfun(@(x) strcmpi(lower(x), allROIs{ROIind}), variableNames,'UniformOutput',false)));
                            leftMat(paInd, 2, ROIind) = allFeatureMat(1,dataId,ind);
                            
                        end
                        
                    elseif strcmp(currentHand,'right')
                        for ROIind = 1:size(allROIs,2)
                            
                            dataId = find(cellfun(@(x) x==1, cellfun(@(x) strcmpi(lower(x), allROIs{ROIind}), variableNames,'UniformOutput',false)));
                            rightMat(paInd, 2, ROIind) = allFeatureMat(1,dataId,ind);
                            
                        end
                        
                    else
                        error('Error with hand side');
                    end
                    
                    
                end
                
            end
            
        end
        
    end
    %% Plot the data
    if strfind(curCharTitle, 'Annova')
        for ROIind = 1:size(allROIs,2)
            %% Left Hand
            titleTxt = ['Left Hand : ' curCharTitle ' Vs. ' allROIs{ROIind}];
            [a,b,stats]=anova1(leftMat(:,2,ROIind), leftMat(:,1,ROIind));
            [c,~,~,gnames] = multcompare(stats);
            compTable = array2table(c(:,[1,2,6]), 'VariableName', {'group 1','group 2', 'p_value'});
            close all
            L_Fig = figure(1);
            axes('position',[.1,.4,.8,.5])
            boxplot(leftMat(:,2,ROIind), leftMat(:,1,ROIind));
            ylabel(currentAnalysis);
            txtBox = sprintf('%s',titleTxt);
            title(txtBox, 'Interpreter', 'none');

            % Convert Table to cell to char array
            tableCell = [compTable.Properties.VariableNames; table2cell(compTable)];
            tableCell(cellfun(@isnumeric,tableCell)) = cellfun(@num2str, tableCell(cellfun(@isnumeric,tableCell)),'UniformOutput',false);
            tableChar = splitapply(@strjoin,pad(tableCell),[1:size(tableCell,1)]');
            % Add axes (not visible) & text (use a fixed width font)
            axes('position',[.1,.15,.8,.2], 'Visible','off');
            text(.2,.95,tableChar,'VerticalAlignment','Top','HorizontalAlignment','Left','FontName','Consolas', 'Interpreter', 'none');
            
            saveas(L_Fig, [savepath 'Left\' curCharTitle ' _Vs_ ' allROIs{ROIind} '.png']);
            close all
            
            %% Right Hand
            titleTxt = ['Left Hand : ' curCharTitle ' Vs. ' allROIs{ROIind}];
            [a,b,stats]=anova1(rightMat(:,2,ROIind), rightMat(:,1,ROIind));
            [c,~,~,gnames] = multcompare(stats);
            compTable = array2table(c(:,[1,2,6]), 'VariableName', {'group 1','group 2', 'p_value'});
            close all
            L_Fig = figure(1);
            axes('position',[.1,.4,.8,.5])
            boxplot(rightMat(:,2,ROIind), rightMat(:,1,ROIind));
            ylabel(currentAnalysis);
            txtBox = sprintf('%s',titleTxt);
            title(txtBox, 'Interpreter', 'none');
            
            % Convert Table to cell to char array
            tableCell = [compTable.Properties.VariableNames; table2cell(compTable)];
            tableCell(cellfun(@isnumeric,tableCell)) = cellfun(@num2str, tableCell(cellfun(@isnumeric,tableCell)),'UniformOutput',false);
            tableChar = splitapply(@strjoin,pad(tableCell),[1:size(tableCell,1)]');
            % Add axes (not visible) & text (use a fixed width font)
            axes('position',[.1,.15,.8,.2], 'Visible','off');
            text(.2,.95,tableChar,'VerticalAlignment','Top','HorizontalAlignment','Left','FontName','Consolas', 'Interpreter', 'none');
            
            saveas(L_Fig, [savepath 'Right\' curCharTitle ' _Vs_ ' allROIs{ROIind} '.png']);
            close all
            
        end
    else
        for ROIind = 1:size(allROIs,2)
            %% Left Hand
            L_Fig = figure(1);
            titleTxt = ['Left Hand : ' curCharTitle ' Vs. ' allROIs{ROIind}];
            
            plot(leftMat(:,1,ROIind), leftMat(:,2,ROIind), 'o', 'color', [0 0.4470 0.7410],...
                'MarkerSize', 10);
            mdl = fitlm(leftMat(:,1,ROIind),leftMat(:,2,ROIind));
            txtBox = sprintf('%s\nR^2= %0.2f,                   R^2 Adjusted= %0.02f\nMSE= %0.02f,                         \tP-val= %0.02f',...
                titleTxt,mdl.Rsquared.Ordinary, mdl.Rsquared.Adjusted, mdl.MSE, mdl.Coefficients.pValue(2));
            %      annotation('textbox',[0.75, 0.8,0.1,0.1],'String', txtBox );
            title(txtBox, 'Interpreter', 'none');
            xlabel(curCharTitle, 'Interpreter', 'none');
            ylabel(currentAnalysis, 'Interpreter', 'none');
            
            saveas(L_Fig, [savepath 'Left\' curCharTitle ' _Vs_ ' allROIs{ROIind} '.png']);
            
            %% Right Hand
            R_Fig = figure(2);
            titleTxt = ['Right Hand : ' curCharTitle ' Vs. ' allROIs{ROIind}];
            mdl = fitlm(rightMat(:,1,ROIind),rightMat(:,2,ROIind));
            txtBox = sprintf('%s\nR^2= %0.2f,                   R^2 Adjusted= %0.02f\nMSE= %0.02f,                         \tP-val= %0.02f',...
                titleTxt,mdl.Rsquared.Ordinary, mdl.Rsquared.Adjusted, mdl.MSE, mdl.Coefficients.pValue(2));
            plot(rightMat(:,1,ROIind), rightMat(:,2,ROIind), 'o', 'color', [0 0.4470 0.7410],...
                'MarkerSize', 10);
            title(txtBox, 'Interpreter', 'none');
            xlabel(curCharTitle, 'Interpreter', 'none');
            ylabel(currentAnalysis, 'Interpreter', 'none');
            saveas(R_Fig, [savepath 'Right\' curCharTitle ' _Vs_ ' allROIs{ROIind} '.png']);
            close all
            
        end
    end
end
