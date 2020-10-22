function verify_full_rank_condition(num_of_agents,num_of_samples,H_c,H_a,gamma_c,gamma_a,sigma,P)
    cvx_begin sdp quiet
    variable X_1(num_of_agents,num_of_agents) hermitian, 
    variable X_2(num_of_agents,num_of_agents) hermitian, 
    variable lambda(3);
    maximize(lambda(3));
    subject to         
    real(trace(H_c*X_1)) >= gamma_c*real(trace(H_c*X_2))+gamma_c*sigma;  
    for j=1:num_of_samples  
        real(trace(H_a(:,:,j)*X_1)) <= gamma_a*real(trace(H_a(:,:,j)*X_2))+gamma_a*sigma; 
    end
     for i=1:num_of_agents
         X_1(i,i)+X_2(i,i) <= P; 
     end
    X_1 >= lambda(1)*eye(num_of_agents);
    X_2 >= lambda(2)*eye(num_of_agents);
    lambda(3) <= lambda(1);
    lambda(3) <= lambda(2);
    X_1 == hermitian_semidefinite(num_of_agents);
    X_2 == hermitian_semidefinite(num_of_agents);
    cvx_end
    if ismember(NaN,X_1) || ismember(NaN,X_2)
        disp('ERROR: The optimization problem is infeasible.')
    end
    if lambda(3) == 0
        disp('WARNING: Slater condition does not hold for the problem instance')
    end
end