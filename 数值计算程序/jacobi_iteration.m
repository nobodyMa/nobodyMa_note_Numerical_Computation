function [x, iter, error_history] = jacobi_iteration(A, b, x0, max_iter, tol)
% 使用矩阵形式的Jacobi迭代法求解线性方程组 Ax = b
%
% 输入:
%   A - 系数矩阵 (n x n)
%   b - 右端向量 (n x 1)
%   x0 - 初始猜测向量 (n x 1)
%   max_iter - 最大迭代次数 (默认: 1000)
%   tol - 容差 (默认: 1e-6)
%
% 输出:
%   x - 数值解
%   iter - 实际迭代次数
%   error_history - 每次迭代的误差历史

    if nargin < 4
        max_iter = 1000;
    end
    if nargin < 5
        tol = 1e-6;
    end
    
    n = length(b);
    x = x0;
    error_history = zeros(max_iter, 1);
    
    % 提取对角矩阵D
    D = diag(diag(A));
    
    % 计算迭代矩阵 R = I - D^(-1)A 和向量 g = D^(-1)b
    R = eye(n) - D \ A;
    g = D \ b;
    
    for iter = 1:max_iter
        % Jacobi迭代: x(k) = R * x(k-1) + g
        x_new = R * x + g;
        
        % 计算相对误差
        error = norm(x_new - x, inf) / norm(x_new, inf);
        error_history(iter) = error;
        
        % 更新解
        x = x_new;
        
        % 检查收敛
        if error < tol
            break;
        end
    end
    
    % 截断误差历史数组
    error_history = error_history(1:iter);
end