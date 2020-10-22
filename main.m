clear all
clc


%% Parameters

r_C = 300; % client's distance
r_b = 80; % radius of the disc that contains the agents
phi_C = 0; % client's polar coordinate
r_A = 300; % the distance of adversaries
c = 299792458; % speed of light
f_c = 4e7; % carrier frequency 40 MHz
lambda_c = c/f_c; % effective wavelength

num_of_agents = 6; % total number of agents
num_of_adversaries=3; % total number of adversaries

adv_sector=zeros(num_of_adversaries,2);
adv_sector(1,:)=[pi/6,4*pi/6]; % directions for adversary a_1
adv_sector(2,:)=[4*pi/6,7*pi/6]; % directions for adversary a_2
adv_sector(3,:)=[7*pi/6,11*pi/6]; % directions for adversary a_3

num_of_samples=1000; % number of samples for adversary locations
                     % this number is denoted by B in the paper                   
sigma=1; % variance of the gaussian noise
gamma_c=10; % client's SINR should be above 10 dBs
gamma_a=1; % adversaries' decibel should be below 0 dBs
P=1; % Maximum power of the agents' antennas

%% Construct the problem instance and synthesize the beamformer


% Randomly generated agent locations in polar coordinates 
agent_dist = r_b .* rand(num_of_agents,1); 
agent_ang = 2*pi .* rand(num_of_agents,1); 

% SINR matrix holds SINR information for all possible directions 
% when the beamforming is performed using the solution of the SDP
angles=-pi/6:0.001:2*pi-pi/6-0.001;
SINR=zeros(num_of_adversaries,size(angles,2));

% We form and solve the SDP relaxation for each adversary
for adv_iter=1:num_of_adversaries
    
    % Randomly sampled adversary locations from the uncertainty inverval
    % I_i, i.e., adv_sector(i,:)
    adv_dist= r_A .* ones(num_of_samples,1);
    adv_ang = adv_sector(adv_iter,1) + (adv_sector(adv_iter,2)-...
              adv_sector(adv_iter,1)).*rand(num_of_samples,1);

    % Channels to the client and randomly sampled adversaries
    % Channel is assumed to perfect and have the form e^{i * 2pi/lambda * distance }
    channel_to_client = zeros(num_of_agents,1);
    channel_to_adv = zeros(num_of_agents,num_of_samples);
    for i=1:num_of_agents
        dist_to_client = r_C-agent_dist(i)*cos(agent_ang(i)-phi_C);
        channel_to_client(i) = exp(1i*(2*pi/lambda_c*dist_to_client));
        for j=1:num_of_samples
             dist_to_adv = r_A-agent_dist(i)*cos(agent_ang(i)-adv_ang(j));
             channel_to_adv(i,j) = exp(1i*(2*pi/lambda_c*dist_to_adv));
        end
    end

    % Channel matrices H to the client and the adversaries
    H_c=channel_to_client(1:num_of_agents)*channel_to_client(1:num_of_agents)';

    H_a=zeros(num_of_agents,num_of_agents,num_of_samples);
    for j=1:num_of_samples
        H_a(:,:,j)=channel_to_adv(1:num_of_agents,j)*channel_to_adv(1:num_of_agents,j)';
    end
    
    % Verify the Slater's condition for the corresponding SDP using CVX solver
    verify_full_rank_condition(num_of_agents,num_of_samples,H_c,H_a,gamma_c,gamma_a,sigma,P)
    % Solve the corresponding SDP using CVX solver
    [X_1,X_2]=SDP_based_beamformer(num_of_agents,num_of_samples,H_c,H_a,gamma_c,gamma_a,sigma,P);

    %%%% Extract weight vectors
    if rank(X_1,1e-5) > 1
        fprintf('ERROR: Beamformer matrix is supposed to be rank 1')
    end
    [eig_vec1,eig_val1]=eig(X_1);
    [max_value,max_index]=max(eig(X_1));
    w_1=sqrt(max_value)*eig_vec1(:,max_index);

    %%% Extract SINR data for all directions
    for i=1:size(SINR,2)
        channel=zeros(num_of_agents,1);
        dist=zeros(num_of_agents,1);
        for j=1:num_of_agents
            dist(j) = r_A-agent_dist(j)*cos(agent_ang(j)-angles(i));
            channel(j) = exp(1i*(2*pi/lambda_c*dist(j)));
        end
        H_num=channel(1:num_of_agents)*channel(1:num_of_agents)';
        SINR(adv_iter,i) = real((w_1'*H_num*w_1))/real((trace(H_num*X_2)+sigma));
    end
end

%% PLOTS
figure % First time step
ax=gca;
plot(angles*(360/(2*pi)),SINR(1,:),'LineWidth',2)
hold on
plot(angles*(360/(2*pi)),ones(length(angles),1),'r','LineStyle',...
    '--','LineWidth',2)
plot(angles*(360/(2*pi)),10*ones(length(angles),1),'r','LineStyle',...
    '--','LineWidth',2)
plot(ones(length(0:0.01:10),1)*adv_sector(1,1)*(360/(2*pi)),...
    0:0.01:10,'r','LineStyle','--','LineWidth',2)
plot(ones(length(0:0.01:10),1)*adv_sector(1,2)*(360/(2*pi)),...
    0:0.01:10,'r','LineStyle','--','LineWidth',2)
xlim([-30,330]);
ylim([0,11]);
xticks([-30:30:330]);
ax.TickLabelInterpreter = 'latex';
xticklabels({'$-\frac{\pi}{6}$','$0$','$\frac{\pi}{6}$','$\frac{2\pi}{6}$',...
              '$\frac{3\pi}{6}$','$\frac{4\pi}{6}$','$\frac{5\pi}{6}$',...
             '$\pi$','$\frac{7\pi}{6}$','$\frac{8\pi}{6}$','$\frac{9\pi}{6}$',...
             '$\frac{10\pi}{6}$','$\frac{11\pi}{6}$'});
yticks([0:1:11]);
message=sprintf(' Possible \n directions for \n adversary a_1');
text(32,3.2,message,'FontSize',15)
annotation('textarrow', [1.7/6 1.3/6],[8.3/11 8/11],'String','Client',...
    'LineWidth',2,'FontSize',14)
annotation('arrow', [1.7/6 1.7/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [1.9/6 1.9/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [2.1/6 2.1/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [2.3/6 2.3/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [2.5/6 2.5/6],[3/11 2.2/11],'LineWidth',2)
ax.FontSize=18;
xlabel('Far-field directions \theta (radians)')
ylabel('SINR_1(\theta)')
title('The beampattern of the pair $({\textbf{w}}^{\star}_1,\Sigma^{\star}_1)$','Interpreter','latex')
grid on

figure % Second time step
ax=gca;
plot(angles*(360/(2*pi)),SINR(2,:),'LineWidth',2)
hold on
plot(angles*(360/(2*pi)),ones(length(angles),1),'r','LineStyle',...
    '--','LineWidth',2)
plot(angles*(360/(2*pi)),10*ones(length(angles),1),'r','LineStyle',...
    '--','LineWidth',2)
plot(ones(length(0:0.01:10),1)*adv_sector(2,1)*(360/(2*pi)),...
    0:0.01:10,'r','LineStyle','--','LineWidth',2)
plot(ones(length(0:0.01:10),1)*adv_sector(2,2)*(360/(2*pi)),...
    0:0.01:10,'r','LineStyle','--','LineWidth',2)
xlim([-30,330]);
ylim([0,11]);
xticks([-30:30:330]);
ax.TickLabelInterpreter = 'latex';
xticklabels({'$-\frac{\pi}{6}$','$0$','$\frac{\pi}{6}$','$\frac{2\pi}{6}$',...
              '$\frac{3\pi}{6}$','$\frac{4\pi}{6}$','$\frac{5\pi}{6}$',...
             '$\pi$','$\frac{7\pi}{6}$','$\frac{8\pi}{6}$','$\frac{9\pi}{6}$',...
             '$\frac{10\pi}{6}$','$\frac{11\pi}{6}$'});
yticks([0:1:11]);
ax=gca;
message=sprintf(' Possible \n directions for \n adversary a_2');
text(122,3.2,message,'FontSize',15)
annotation('textarrow', [1.7/6 1.3/6],[8.3/11 8/11],'String','Client',...
    'LineWidth',2,'FontSize',14)
annotation('arrow', [2.8/6 2.8/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [3/6 3/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [3.2/6 3.2/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [3.4/6 3.4/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [3.6/6 3.6/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [3.8/6 3.8/6],[3/11 2.2/11],'LineWidth',2)
ax.FontSize=18;
xlabel('Far-field directions \theta (radians)')
ylabel('SINR_2(\theta)')
title('The beampattern of the pair $({\textbf{w}}^{\star}_2,\Sigma^{\star}_2)$','Interpreter','latex')
grid on

figure % Third time step
ax=gca;
plot(angles*(360/(2*pi)),SINR(3,:),'LineWidth',2)
hold on
plot(angles*(360/(2*pi)),ones(length(angles),1),'r','LineStyle',...
    '--','LineWidth',2)
plot(angles*(360/(2*pi)),10*ones(length(angles),1),'r','LineStyle',...
    '--','LineWidth',2)
plot(ones(length(0:0.01:10),1)*adv_sector(3,1)*(360/(2*pi)),...
    0:0.01:10,'r','LineStyle','--','LineWidth',2)
plot(ones(length(0:0.01:10),1)*(adv_sector(3,2)*(360/(2*pi))-2),...
    0:0.01:10,'r','LineStyle','--','LineWidth',2)
xlim([-30,330]);
ylim([0,11]);
xticks([-30:30:330]);
ax.TickLabelInterpreter = 'latex';
xticklabels({'$-\frac{\pi}{6}$','$0$','$\frac{\pi}{6}$','$\frac{2\pi}{6}$',...
              '$\frac{3\pi}{6}$','$\frac{4\pi}{6}$','$\frac{5\pi}{6}$',...
             '$\pi$','$\frac{7\pi}{6}$','$\frac{8\pi}{6}$','$\frac{9\pi}{6}$',...
             '$\frac{10\pi}{6}$','$\frac{11\pi}{6}$'});
yticks([0:1:11]);
message=sprintf(' Possible \n directions for \n adversary a_3');
text(232,3.2,message,'FontSize',15)
annotation('textarrow',[1.7/6 1.3/6],[8.3/11 8/11],'String','Client',...
    'LineWidth',2,'FontSize',14)
annotation('arrow', [4/6 4/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [4.2/6 4.2/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [4.4/6 4.4/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [4.6/6 4.6/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [4.8/6 4.8/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [5/6 5/6],[3/11 2.2/11],'LineWidth',2)
annotation('arrow', [5.2/6 5.2/6],[3/11 2.2/11],'LineWidth',2)
ax.FontSize=18;
xlabel('Far-field directions \theta (radians)')
ylabel('SINR_3(\theta)')
title('The beampattern of the pair $({\textbf{w}}^{\star}_3,\Sigma^{\star}_3)$','Interpreter','latex')
grid on