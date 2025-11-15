function [x, iter, error_history] = gauss_seidel_iteration(A, b, x0, max_iter, tol)
% 使用矩阵形式的Gauss-Seidel迭代法求解线性方程组 Ax = b
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
    
    % 矩阵分解: A = D + L + U
    D = diag(diag(A));          % 对角部分
    L = tril(A, -1);           % 严格下三角部分
    U = triu(A, 1);            % 严格上三角部分
    
    % 计算迭代矩阵 G = -(D+L)^(-1)U 和向量 g = (D+L)^(-1)b
    % 注意: 实际计算中避免直接求逆，使用前向替换
    D_sum_L = D + L;
    
    for iter = 1:max_iter
        x_old = x;
        
        % Gauss-Seidel迭代: (D+L)x(k) = -Ux(k-1) + b
        % 使用前向替换求解 (D+L)x(k) = b - U*x_old
        
        rhs = b - U * x_old;
        
        % 前向替换求解下三角系统
        x = zeros(n, 1);
        for i = 1:n
            x(i) = (rhs(i) - D_sum_L(i, 1:i-1) * x(1:i-1)) / D_sum_L(i, i);
        end
        
        % 计算相对误差
        error = norm(x - x_old, inf) / norm(x, inf);
        error_history(iter) = error;
        
        % 检查收敛
        if error < tol
            break;
        end
    end
    
    % 截断误差历史数组
    error_history = error_history(1:iter);
end
