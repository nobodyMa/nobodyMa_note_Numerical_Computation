function [solution, iterations, errors] = newton_system(f, vars, x0, max_iter, tol)
    % 使用 Newton 迭代法求解非线性方程组
    % 
    % 输入:
    %   f - 符号函数向量
    %   vars - 符号变量向量  
    %   x0 - 初始猜测
    %   max_iter - 最大迭代次数 (默认: 50)
    %   tol - 容差 (默认: 1e-12)
    % 
    % 输出:
    %   solution - 数值解
    %   iterations - 实际迭代次数
    %   errors - 每次迭代的误差历史
    
    if nargin < 4
        max_iter = 50;
    end
    if nargin < 5
        tol = 1e-12;
    end
    
    % 计算 Jacobian 矩阵
    J = jacobian(f, vars);
    
    % 转换为数值函数句柄
    f_func = matlabFunction(f, 'Vars', {vars});
    J_func = matlabFunction(J, 'Vars', {vars});
    
    x = x0;
    errors = zeros(max_iter, 1);
    
    for iterations = 1:max_iter
        % 计算当前函数值和 Jacobian
        f_val = f_func(x);
        J_val = J_func(x);
        
        % 计算误差
        current_error = norm(f_val);
        errors(iterations) = current_error;
        
        % 检查收敛
        if current_error < tol
            break;
        end
        
        % Newton 迭代
        delta_x = -J_val \ f_val;
        x = x + delta_x;
        
        % 检查步长是否太小
        if norm(delta_x) < 1e-14
            disp('delta_x 小于1e-14')
            break;
        end
    end

    solution = x;
end