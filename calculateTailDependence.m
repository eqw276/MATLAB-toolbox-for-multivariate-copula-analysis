function [lambda_upper, lambda_lower] = calculateTailDependence(copula_family, params)
% 计算Copula的上下尾部依赖系数
% 输入:
%   copula_family - Copula家族名称
%   params - Copula参数
% 输出:
%   lambda_upper - 上尾依赖系数
%   lambda_lower - 下尾依赖系数

lambda_upper = NaN;
lambda_lower = NaN;

try
    switch lower(copula_family)
        case 'gaussian'
            % Gaussian Copula: 无尾部依赖
            if length(params) >= 1
                rho = params(1);
                lambda_upper = 0;
                lambda_lower = 0;
            end
            
        case 't'
            % t Copula: 对称的尾部依赖
            if length(params) >= 2
                rho = params(1);
                nu = params(2);
                if nu > 0
                    lambda_upper = 2 * tcdf(-sqrt((nu+1)*(1-rho)/(1+rho)), nu+1);
                    lambda_lower = lambda_upper; % t Copula对称
                end
            end
            
        case 'clayton'
            % Clayton Copula: 只有下尾依赖
            if length(params) >= 1
                theta = params(1);
                if theta > 0
                    lambda_lower = 2^(-1/theta);
                    lambda_upper = 0;
                end
            end
            
        case 'gumbel'
            % Gumbel Copula: 只有上尾依赖
            if length(params) >= 1
                theta = params(1);
                if theta >= 1
                    lambda_upper = 2 - 2^(1/theta);
                    lambda_lower = 0;
                end
            end
            
        case 'frank'
            % Frank Copula: 无尾部依赖
            if length(params) >= 1
                lambda_upper = 0;
                lambda_lower = 0;
            end
            
        case 'joe'
            % Joe Copula: 只有上尾依赖
            if length(params) >= 1
                theta = params(1);
                if theta >= 1
                    lambda_upper = 2 - 2^(1/theta);
                    lambda_lower = 0;
                end
            end
            
        case 'galambos'
            theta = params(1);
            lambda_upper  = 2 - 2^(1/theta);
            lambda_lower  = 0;       
            
        case 'bb1'
            % BB1 Copula: 同时有上下尾依赖
            if length(params) >= 2
                theta = params(1);
                delta = params(2);
                if theta > 0 && delta >= 1
                    lambda_lower = 2^(-1/(theta*delta));
                    lambda_upper = 2 - 2^(1/delta);
                end
            end
        
        case 'bb5'
            % BB5 Copula: 同时有上下尾依赖
            if length(params) >= 2
                theta = params(1);
                delta = params(2);
                if theta >= 1 && delta > 0
                    lambda_upper = 2 - 2^(1/theta);
                    lambda_lower = 2^(-1/(theta*delta));
                end
            end
            
        case 'tawn'
            % Tawn Copula: 非对称尾部依赖
            if length(params) >= 3
                alpha = params(1);
                beta = params(2);
                theta = params(3);
                if theta >= 1
                    lambda_upper = beta * (2 - 2^(1/theta));
                    lambda_lower = alpha * (2 - 2^(1/theta));
                end
            end
            
        case 'independence'
            lambda_upper  = 0; lambda_lower = 0;
            
        case 'amh'
            lambda_upper = 0; lambda_lower = 0;

        case 'fgm'
            lambda_upper  = 0; lambda_lower = 0;

        case 'plackett'
            lambda_upper  = 0; lambda_lower = 0;

        
        otherwise
            % 对于其他Copula，使用数值方法估计
            [lambda_upper, lambda_lower] = numericalTailDependence(copula_family, params);
    end
    
catch err
    fprintf('尾部依赖计算错误 for %s: %s\n', copula_family, err.message);
    % 使用数值方法作为备选
    try
        [lambda_upper, lambda_lower] = numericalTailDependence(copula_family, params);
    catch
        lambda_upper = NaN;
        lambda_lower = NaN;
    end
end

% 确保系数在合理范围内
lambda_upper = max(0, min(1, lambda_upper));
lambda_lower = max(0, min(1, lambda_lower));

end

function [lambda_upper, lambda_lower] = numericalTailDependence(copula_family, params)
% 数值方法估计尾部依赖系数
%
% INPUT:
%   copula_family - Copula类型 (string)
%   params        - Copula参数 (向量)
%
% OUTPUT:
%   lambda_upper  - 上尾依赖系数
%   lambda_lower  - 下尾依赖系数

lambda_upper = NaN;
lambda_lower = NaN;

try
    % -------- 参数设置 --------
    n_samples = 1e5;         % 固定采样数
    u_values = 0.99:0.001:0.999;
    l_values = 0.001:0.001:0.01;
    min_count = 20;          % 分母阈值，避免除以太小的数
    
    % -------- 生成Copula样本 --------
    switch lower(copula_family)
        case 'gaussian'
            Rho = [1, params(1); params(1), 1];
            U = copularnd('Gaussian', Rho, n_samples);
        case 't'
            Rho = [1, params(1); params(1), 1];
            U = copularnd('t', Rho, params(2), n_samples);
        otherwise
            % 其他Copula采用自定义采样
            U = copulaRandomSample(copula_family, params, n_samples);
    end
    
    % -------- 上尾依赖 --------
    upper_dep = nan(size(u_values));
    for i = 1:length(u_values)
        u = u_values(i);
        denom = sum(U(:,2) > u);
        if denom > min_count
            upper_dep(i) = sum(U(:,1) > u & U(:,2) > u) / denom;
        end
    end
    valid_idx = ~isnan(upper_dep);
    if sum(valid_idx) >= 3
        p = polyfit(u_values(valid_idx), upper_dep(valid_idx), 1);
        lambda_upper = polyval(p, 1);
    end
    
    % -------- 下尾依赖 --------
    lower_dep = nan(size(l_values));
    for i = 1:length(l_values)
        u = l_values(i);
        denom = sum(U(:,2) < u);
        if denom > min_count
            lower_dep(i) = sum(U(:,1) < u & U(:,2) < u) / denom;
        end
    end
    valid_idx = ~isnan(lower_dep);
    if sum(valid_idx) >= 3
        p = polyfit(l_values(valid_idx), lower_dep(valid_idx), 1);
        lambda_lower = polyval(p, 0);
    end
    
catch
    warning('Tail dependence estimation failed for %s copula.', copula_family);
end

end

function U = copulaRandomSample(copula_family, params, n)
% 通用Copula随机样本生成（可扩展）
try
    switch lower(copula_family)
        case {'clayton', 'frank', 'gumbel'}
            U = copularnd(copula_family, params(1), n);
        otherwise
            % 占位：若未实现，先返回独立样本
            U = rand(n, 2);
    end
catch
    U = rand(n, 2);
end
end