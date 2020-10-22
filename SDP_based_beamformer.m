function [X_1,X_2]=SDP_based_beamformer(num_of_agents,num_of_samples,H_c,H_a,gamma_c,gamma_a,sigma,P)
    cvx_begin sdp quiet
    variable X_1(num_of_agents,num_of_agents) hermitian, 
    variable X_2(num_of_agents,num_of_agents) hermitian, 
    minimize(trace(X_1)+trace(X_2));
    subject to         
    real(trace(H_c*X_1)) >= gamma_c*real(trace(H_c*X_2))+gamma_c*sigma;  
    for j=1:num_of_samples  
        real(trace(H_a(:,:,j)*X_1)) <= gamma_a*real(trace(H_a(:,:,j)*X_2))+gamma_a*sigma; 
    end
     for i=1:num_of_agents
         X_1(i,i)+X_2(i,i) <= P; 
     end
    X_1 == hermitian_semidefinite(num_of_agents);
    X_2 == hermitian_semidefinite(num_of_agents);
    cvx_end
    if isnan(real(X_1(1,1))) || isnan(real(X_2(1,1)))
        disp('ERROR: The optimization problem is infeasible.')
    end
end