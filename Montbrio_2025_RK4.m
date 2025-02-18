% Replication of results from paper "Low-dimensional model for adaptive
% networks of spiking neurons" by  B. Pietras, P. Clusella and E. Montbrió 
% https://doi.org/10.1103/PhysRevE.111.014422
%
% Here we compare results from integration of the network of theta [QIF] neurons
% vs solutions of macroscopic equations for spiking rate, mean potential
% and adaptation variable.
% Integration is performed with Runge-Kutta algorithm
%
    close all
    clear all
    
    % Number of theta neurons
    N= 100000;

    % Neuron parameters
    eta = 0.5; % center of Lorenz distribution
    delta = 1;
    J = 20;
    taua = 100;
    taum = 10;
    beta = 1;

    j1=1:N;
    % eta values for individual neurons
    etam = (eta+delta*tan(pi/2*(2*j1-N-1)/(N+1)))';

    %initial values of the macroscopic variables
    V0 = 0;
    r0 = 0.1;
    %initial potential of every neuron
    v0 = (V0+r0*pi*taum*tan(pi/2*(2*j1-N-1)/(N+1)))'; 

    %potential converted to theta variable
    tet = 2*atan(v0);
    aj = zeros(N,1);

    P.N    =   N; 
    P.J    =   J; 
    P.etam = etam;
    P.beta = beta;
    P.taum = taum;
    P.taua = taua;
    P.etab = eta;
    P.delta = delta;

    %integration time
    Tend= 100;
    %integration step
    dt = 1e-3;
    %number of integration steps
    Nt = round(Tend/dt)+1;
    
    dM = 2*dt;

    tm = linspace(0,(Nt-1)*dt,Nt);
    %arrays for storing Kuramoto order parameter 
    Zm = zeros(Nt,1);
    % mean adaptaion variable
    Aj = zeros(Nt,1);
    % solution of macroscopic equations
    Y = zeros(Nt,3);
       
   
    %initial condition for the microscopic equations
    y = [tet;aj];


    t = 0;
    tic
    fprintf('Beggining of microscopic equations integration\n')
    for i1=1:Nt
        t = t+dt;
        if mod(i1/(Nt-1)*100,10)==0
            fprintf("%f proc.\n",i1/(Nt-1)*100)
            toc
        end

        Zm(i1) = mean(exp(1i*y(1:N)));
        Aj(i1) = mean(y(N+1:2*N));

        y =rk4(@(t,y)eqsns_micro(t,y,P),t,y,dt);
        y(1:N) = mod(y(1:N),2*pi);

    end
    toc

    fprintf('End of microscopic equations integration\n\n\n')

    % mean potential and spiking rates estimated from Kuramoto order
    % parameter
    v =imag((1-conj(Zm))./(1+conj(Zm)));
    r =1/(pi*taum)*real((1-conj(Zm))./(1+conj(Zm)));
   
    
    figure(1)
    subplot(311)
    plot(tm,r), hold on
    ylabel('r')
    subplot(312)
    plot(tm,v), hold on
    ylabel('v')
    subplot(313)
    plot(tm,Aj), hold on
    ylabel('A')

    %initial condition for the macroscopic equations
    y = [r0; V0; 0];
    t = 0;
    tic
    for i1=1:Nt
        t = t+dt;
        if mod(i1/(Nt-1)*100,10)==0
            fprintf("%f proc.\n",i1/(Nt-1)*100)            
        end

        Y(i1,:) = y;
        y =rk4(@(t,y)eqsns_macro(t,y,P),t,y,dt);

    end
    toc


    figure(1)
    subplot(311)
    plot(tm,Y(:,1)), hold on
    ylabel('r')
    subplot(312)
    plot(tm,Y(:,2)), hold on
    ylabel('v')
    subplot(313)
    plot(tm,Y(:,3)), hold on
    ylabel('A')



    

function y = rk4(F_xy,t,y,h)   
    k_1 = F_xy(t,y);
    k_2 = F_xy(t+0.5*h,y+0.5*h*k_1);
    k_3 = F_xy((t+0.5*h),(y+0.5*h*k_2));
    k_4 = F_xy((t+h),(y+k_3*h));
    y = y + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;  % main equation
end

function dy = eqsns_micro(t,y,P)  
% Microscopic network simulations of QIF neurons with
% quadratic spike-frequency adaptation 
        N    = P.N;
        J    = P.J;
        etam = P.etam;
        beta = P.beta;
        taum = P.taum;
        taua = P.taua;

        Zm = mean(exp(1i*y(1:N)));
        Zb= conj(Zm);
        W = (1-Zb)./(1+Zb);   
        R  = 1/(pi*taum)*real(W);
        
        cs=cos(y(1:N));
        Fn = (etam - y(N+1:2*N) + J*taum*R);

        dy = zeros(2*N,1);
        dy(1:N)     = ( 1-cs+(1+cs).*Fn )/taum;
        dy(N+1:2*N) = ( -y(N+1:2*N) +   beta*Fn )/taua;
end



function dy = eqsns_macro(t,y,P)
% Macroscopic equations defining behaviour of the network of QIF neurons with
% quadratic spike-frequency adaptation 

    taum = P.taum;
    etab = P.etab;
    J = P.J;
    delta = P.delta;
    beta = P.beta;
    taua = P.taua;

    dy = zeros(3,1);

    % Spiking Rate equation
    dy(1) = ( delta/((1+beta)*pi*taum)+2*y(1)*y(2))/taum;
    % Mean potential
    dy(2) = ( y(2)^2 - (taum*pi*y(1))^2 + etab + J *taum*y(1)-y(3) )/taum;
    % Adaptation variable
    dy(3) = (-y(3)*(1+beta)+beta*(etab+J*taum*y(1)))/taua;

end

