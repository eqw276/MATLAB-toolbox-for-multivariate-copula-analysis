clc; clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% MATLAB code developed by Mojtaba Sadegh (mojtabasadegh@gmail.com) and Amir AghaKouchak,
% Center for Hydrometeorology and Remote Sensing (CHRS)
% University of California, Irvine
%
% Last modified on December 20, 2016
%
% Please reference to:
% Sadegh, M., E. Ragno, and A. AghaKouchak (2017), MvDAT: Multivariate
% Dependence Analysis Toolbox, Water Resources Research, 53, doi:10.1002/2016WR020242.
% Link: http://onlinelibrary.wiley.com/doi/10.1002/2016WR020242/epdf
%
% Please contact Mojtaba Sadegh (mojtabasadegh@gmail.com) with any issue.
%
% Disclaimer:
% This program (hereafter, software) is designed for instructional, educational and research use only.
% Commercial use is prohibited. The software is provided 'as is' without warranty
% of any kind, either express or implied. The software could include technical or other mistakes,
% inaccuracies or typographical errors. The use of the software is done at your own discretion and
% risk and with agreement that you will be solely responsible for any damage and that the authors
% and their affiliate institutions accept no responsibility for errors or omissions in the software
% or documentation. In no event shall the authors or their affiliate institutions be liable to you or
% any third parties for any special, indirect or consequential damages of any kind, or any damages whatsoever.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% User Input
% Variable names
handles.U1_name = 'Variable 1'; % insert name of variable in the first column
handles.U2_name = 'Variable 2'; % insert name of variable in the first column

% Data file name
datafile ="C:\Users\辛昊阳\Desktop\Copula-based non-stationarity detection of monthly runoff and sediment discharge in karst catchments\二审\数据\按时间分开\yujiang_dry.xlsx";   % 如果文件就在脚本同目录下，直接用文件名即可

handles.data = readmatrix(datafile);

% Default is not to plot empirical probability isolines
handles.EmProbIsLine = 0; % 1: plot empirical prob isolines, 0: don't plot them

% Local optimization or Global
handles.Optimization = 'MCMC'; % Local: local optimization; MCMC: both local & global optimization

% Which Copulas to run? Select any combination
% 1: Gaussian, 2: t, 3: Clayton, 4: Frank, 5: Gumbel, 6: Independence, 7: Ali-Mikhail-Haq (AMH), 8: Joe
% 9: Farlie-Gumbel-Morgenstern (FGM), 10: Plackett, 11: Cuadras-Auge, 12: Raftery
% 13: Shih-Louis, 14: Linear-Spearman, 15: Cubic, 16: Burr, 17: Nelson, 18: Galambos, 19: Marshal-Olkin
% 20: Fischer-Hinzmann, 21: Roch-Alegre, 22: Fischer-Kock, 23: BB1, 24: BB5, 25: Tawn
handles.ID_CHOSEN = [1:8];

% Calculate pvalues or not? 0: no pvalue; 1: pvalue for the best copula
% selected according to BIC; 2: pvalue for all chosen copulas
handles.pvalue = 2;

% Run main MvCAT function
[ID_ML,ID_AIC,ID_BIC,Family] = MvCAT_main2(handles);