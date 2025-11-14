function [solution, converged, iterations, errors, details] = newton_system_solver(f, vars, x0, options)
% 使用 Newton 迭代法求解非线性方程组
%
% 语法:
%   [solution, converged, iterations, errors] = newton_system_solver(f, vars, x0)
%   [solution, converged, iterations, errors, details] = newton_system_solver(f, vars, x0, options)
%
% 输入参数:
%   f       - 符号表达式向量或函数句柄，表示方程组 f(x) = 0
%   vars    - 符号变量向量
%   x0      - 初始猜测向量
%   options - 可选参数结构体，包含：
%             .MaxIterations - 最大迭代次数 (默认: 500)
%             .Tolerance     - 收敛容差 (默认: 1e-12)
%             .Display       - 显示选项: 'none', 'iter', 'final' (默认: 'final')
%             .Jacobian      - Jacobian 计算方式: 'symbolic', 'numeric', 'auto' (默认: 'auto')
%             .StepTolerance - 步长容差 (默认: 1e-14)
%             .FiniteDiffStep - 有限差分步长 (默认: 1e-8)
%
% 输出参数:
%   solution  - 数值解向量
%   converged - 逻辑值，指示是否收敛
%   iterations - 实际迭代次数
%   errors    - 每次迭代的误差历史 ||f(x_k)||_2
%   details   - 包含详细信息的结构体

    % 参数验证和默认值设置
    if nargin < 4
        options = struct();
    end
    
    % 设置默认选项
    default_options = struct(...
        'MaxIterations', 500, ...
        'Tolerance', 1e-12, ...
        'Display', 'final', ...
        'Jacobian', 'auto', ...
        'StepTolerance', 1e-14, ...
        'FiniteDiffStep', 1e-8);
    
    options = merge_options(default_options, options);
    
    % 输入验证
    validateattributes(x0, {'numeric'}, {'vector', 'finite'}, mfilename, 'x0');
    x0 = x0(:); % 确保是列向量
    
    % 确定 Jacobian 计算方式
    if strcmp(options.Jacobian, 'auto')
        if isa(f, 'sym') || isa(f, 'symfun')
            options.Jacobian = 'symbolic';
        else
            options.Jacobian = 'numeric';
        end
    end
    
    % 准备函数句柄和 Jacobian
    [f_func, J_func, using_symbolic] = prepare_functions(f, vars, options);
    
    % 显示迭代信息头
    if strcmp(options.Display, 'iter')
        fprintf('\n%6s %15s %15s %15s\n', 'Iter', '||f(x)||', '||Δx||', 'Cond(J)');
        fprintf('%6s %15s %15s %15s\n', '----', '-----------', '-----------', '-----------');
    end
    
    % 初始化变量
    x = x0;
    n = length(x0);
    errors = zeros(options.MaxIterations, 1);
    steps = zeros(options.MaxIterations, 1);
    condition_numbers = zeros(options.MaxIterations, 1);
    converged = false;
    
    % 主迭代循环
    for iterations = 1:options.MaxIterations
        % 计算当前函数值
        f_val = f_func(x);
        current_error = norm(f_val);
        errors(iterations) = current_error;
        
        % 检查函数值收敛
        if current_error < options.Tolerance
            converged = true;
            break;
        end
        
        % 计算 Jacobian 矩阵
        if using_symbolic
            J_val = J_func(x);
        else
            J_val = numerical_jacobian(f_func, x, options.FiniteDiffStep, n);
        end
        
        % 检查 Jacobian 条件数
        cond_J = cond(J_val);
        condition_numbers(iterations) = cond_J;
        
        if cond_J > 1e5
            warning('Newton:IllConditioned', ...
                'Jacobian 矩阵条件数过大 (%g)，结果可能不可靠', cond_J);
        end
        
        % 求解线性系统：J(x_k) Δx = -f(x_k)
        try
            delta_x = - (J_val \ f_val);
        catch ME
            if ~strcmp(options.Display, 'none')
                fprintf('在迭代 %d 时求解线性系统失败: %s\n', iterations, ME.message);
            end
            break;
        end
        
        step_norm = norm(delta_x);
        steps(iterations) = step_norm;
        
        % 检查步长收敛
        if step_norm < options.StepTolerance
            converged = true;
            break;
        end
        
        % 更新解
        x = x + delta_x;
        
        % 显示迭代信息
        if ~strcmp(options.Display, 'iter')
            fprintf('%6d %15.6e %15.6e %15.6e\n', ...
                iterations, current_error, step_norm, cond_J);
        end
    end
    
    % 截断输出数组
    errors = errors(1:iterations);
    steps = steps(1:iterations);
    condition_numbers = condition_numbers(1:iterations);
    
    solution = x;
    
    % 准备详细信息
    details.iterations = iterations;
    details.final_error = current_error;
    details.final_step = step_norm;
    details.condition_numbers = condition_numbers;
    details.steps = steps;
    details.using_symbolic_jacobian = using_symbolic;
    
    % 显示最终结果
    if ~strcmp(options.Display, 'none')
        display_results(converged, iterations, current_error, using_symbolic);
    end
end

%% 辅助函数

function options = merge_options(defaults, user_options)
% 合并默认选项和用户选项
    if isempty(user_options)
        options = defaults;
        return;
    end
    
    options = defaults;
    user_fields = fieldnames(user_options);
    for i = 1:length(user_fields)
        field = user_fields{i};
        if isfield(defaults, field)
            options.(field) = user_options.(field);
        else
            warning('Newton:UnknownOption', '忽略未知选项: %s', field);
        end
    end
end

function [f_func, J_func, using_symbolic] = prepare_functions(f, vars, options)
% 准备函数句柄和 Jacobian 计算函数
    using_symbolic = false;
    
    if isa(f, 'sym') || isa(f, 'symfun')
        % 符号表达式输入
        using_symbolic = true;
        
        % 转换为数值函数句柄
        f_func = matlabFunction(f, 'Vars', {vars});
        
        if strcmp(options.Jacobian, 'symbolic')
            % 符号计算 Jacobian
            J_sym = jacobian(f, vars);
            J_func = matlabFunction(J_sym, 'Vars', {vars});
        else
            % 数值 Jacobian
            J_func = @(x) numerical_jacobian(f_func, x, options.FiniteDiffStep, length(x));
            using_symbolic = false;
        end
        
    elseif isa(f, 'function_handle')
        % 函数句柄输入，只能使用数值 Jacobian
        f_func = f;
        
        % 必须使用数值 Jacobian
        J_func = @(x) numerical_jacobian(f_func, x, options.FiniteDiffStep, length(x));
        
        % 如果用户要求符号 Jacobian，给出警告
        if strcmp(options.Jacobian, 'symbolic')
            warning('Newton:SymbolicJacobianNotAvailable', ...
                '函数句柄输入无法使用符号 Jacobian，已自动切换到数值 Jacobian');
        end
        
        using_symbolic = false;
        
    else
        error('Newton:InvalidInput', ...
            '输入 f 必须是符号表达式或函数句柄');
    end
end

function J_num = numerical_jacobian(f_func, x, h, n)
% 数值计算 Jacobian 矩阵（中心差分）
% n = length(x);

    f0 = f_func(x);
    m = length(f0);
    J_num = zeros(m, n);
    
    for j = 1:n
        x_forward = x;
        x_backward = x;
        
        x_forward(j) = x_forward(j) + h;
        x_backward(j) = x_backward(j) - h;
        
        f_forward = f_func(x_forward);
        f_backward = f_func(x_backward);
        
        J_num(:, j) = (f_forward - f_backward) / (2 * h);
    end
end

function display_results(converged, iterations, final_error, using_symbolic)
% 显示最终结果
    fprintf('\n=== Newton 迭代法结果 ===\n');
    if converged
        fprintf('收敛成功\n');
    else
        fprintf('未收敛\n');
    end
    fprintf('迭代次数: %d\n', iterations);
    fprintf('最终误差: %.6e\n', final_error);
    if using_symbolic
        fprintf('Jacobian 计算: 符号计算\n');
    else
        fprintf('Jacobian 计算: 数值差分\n');
    end
    fprintf('\n');
end
