%% I will start by the old part of the code.
%% The modifications are from line 60 to line 125
%% // get the grid parameters which remain constants (not time dependent)
% For Ep2 dataset
[H0, W, Grid_Sign] =  initialize_wave( TAG3DSettings.strainwaveparamY ) ;

iframe = 0 ;
for time = (1:CineNoFrames)/3
    %// update wave surface
    Z = calc_wave( H0,W,time,Grid_Sign ) ;
    iframe= iframe+1 ;
    FrameDisplY(:,:,iframe) = Z;%imclose(Z,strel('disk',10)) ;
end
FrameDisplY=smooth3(FrameDisplY,'gaussian',15,15);

%% Simulated Gaussian shape for testing
ShapeSimulation =0;
if ShapeSimulation
    for iframe = 1:CineNoFrames
        FrameDisplX(:,:,iframe)=fspecial('gaussian',256,5+2*iframe);
        FrameDisplY(:,:,iframe)=fspecial('gaussian',256,10+4*iframe);
    end
end
FrameDisplY=-FrameDisplY; % to invert displacement in one drection
%% // Scaling displacement
% Scaling displacement to be capped by MaxFrameDisplY and MaxFrameDisplX
% either in individual timeframe or in the set as a whole.

UseGlobalScaling =0;

if UseGlobalScaling
    MinWave=min(FrameDisplX(:));
    MaxWave=max(FrameDisplX(:));
    FrameDisplX=((FrameDisplX-MinWave)/(MaxWave-MinWave))*(MaxFrameDisplX-MinFrameDisplX)+MinFrameDisplX; % in mm

    MinWave=min(FrameDisplY(:));
    MaxWave=max(FrameDisplY(:));
    FrameDisplY=((FrameDisplY-MinWave)/(MaxWave-MinWave))*(MaxFrameDisplY-MinFrameDisplY)+MinFrameDisplY; % in mm
else
    for iframe = 1:CineNoFrames
        MinWave=min(squeeze(FrameDisplX(:,:,iframe)),[],'all');
        MaxWave=max(squeeze(FrameDisplX(:,:,iframe)),[],'all');
        FrameDisplX(:,:,iframe)=((FrameDisplX(:,:,iframe)-MinWave)/(MaxWave-MinWave))*(MaxFrameDisplX-MinFrameDisplX)+MinFrameDisplX; % in mm
        
        MinWave=min(squeeze(FrameDisplY(:,:,iframe)),[],'all');
        MaxWave=max(squeeze(FrameDisplY(:,:,iframe)),[],'all');
        FrameDisplY(:,:,iframe)=((FrameDisplY(:,:,iframe)-MinWave)/(MaxWave-MinWave))*(MaxFrameDisplY-MinFrameDisplY)+MinFrameDisplY; % in mm
    end
end


if mod(sizeImage,2) ~=0 % odd
    outputXCoord = (-(sizeImage-1)/2 :(sizeImage-1)/2)*deltaX;
    outputYCoord = (-(sizeImage-1)/2 :(sizeImage-1)/2)*deltaY;
else % if even
    outputXCoord = (-(sizeImage)/2 :((sizeImage)/2) - 1)*deltaX;
    outputYCoord = (-(sizeImage)/2 :((sizeImage)/2) - 1)*deltaY;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%BEGIN: Polar Coordinate Modification Idea%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Consider the first output of the wave generator the wave in the radial direction
FrameDisplRads=FrameDisplX; 
%Consider the second output of the wave generator the wave in the theta direction
FrameDisplThta=FrameDisplY;

NX = size(outputXCoord,2); % Num of Cols in the image
NY = size(outputYCoord,2); % Num of Rows in the image

% Form the co-ordinate matrices

xMat = repmat(outputXCoord,NY,1); % X 256x256 grid map in mm
yMat = repmat(outputYCoord,1,NX);% Y 256x256 grid map in mm

deltaT=1; % Time step 1 

OverStrain=1; % excessive strain Flag

% The center of the heart or the origin of the polar coordinates
x0=heartCenter(2)*deltaX; % The x-center of the heart mask, location in mm
y0=heartCenter(1)*deltaY; % The y-center of the heart mask, location in mm

% Creating a new grid with the center at the origin
yMat_shft=yMat - x0; 
xMat_shft=xMat - y0;

% Compute the polar coordinate sgrids R and Theta, centered around the heart
R = sqrt((yMat_shft).^2 + (xMat_shft).^2); % Radial distance
Theta = atan2(yMat_shft, xMat_shft);       % Angle in radians

%Replicate R and Theta maps to all frames assuming the origin is the same
%In the future, more accurate is to use the center of the heart at each
%time-frame
R     = repmat(R    , [1, 1, size(FrameDisplRads,3)]); % Expand to 256x256xNoFrames
Theta = repmat(Theta, [1, 1, size(FrameDisplThta,3)]); % Expand to 256x256xNoFrames

%An optional weighting factor to further tweek the radial/theta displacement near the
%origin
ThetaScale=((R+0.001)/max(R+0.001,[],'all'))*3;
ThetaScale(ThetaScale>2)=2;

%RadiaScale=-(R+eps)/max(R,[],'all')*1;
%RadiaScale(abs(RadiaScale)<0.05)=0.05*sign(RadiaScale(abs(RadiaScale)<0.05));

RadiaScale=-(R+eps)/max(R,[],'all')*1;
RadiaScale(abs(RadiaScale)<0.95)=0.95*sign(RadiaScale(abs(RadiaScale)<0.95));

% Using the old FrameDisplX as the radial direction displacement
%u_r=FrameDisplRads.*ThetaScale+1./RadAddScale;
u_r    =FrameDisplRads./RadiaScale;
% Using the old FrameDisplY as the theta direction displacement
u_theta=FrameDisplThta.*ThetaScale;

% Convert displacements from polar to Cartesian
u_x = u_r .* cos(Theta) - u_theta .* sin(Theta);
u_y = u_r .* sin(Theta) + u_theta .* cos(Theta);

FrameDisplX=u_x;
FrameDisplY=u_y;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%END: Polar Coordinate Modification Idea%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The remaining is the strain calculation
while OverStrain==1
    UX=FrameDisplX; %UX The x-displacement of the material point at this specific spacial point 
    UY=FrameDisplY; %UY The y-displacement of the material point at this specific spacial point 
    
    %DisplxTotal=100*imEp1*deltaX;%smooth3(DisplxTotal,'gaussian',25,2);
    %DisplyTotal=100*imEp2*deltaY;%smooth3(DisplyTotal,'gaussian',25,2);
    
    [UXx, UXy, ExtAll]=gradient(UX,deltaX,deltaY,deltaT);
    [UYx, UYy, EytAll]=gradient(UY,deltaX,deltaY,deltaT);
    
    % Eulerian Strain Calculation based on the the left Cauchy-Green
    % deformation tensor as reported in thesis-khz on pages 21-22
    % Inplane strain tensor E=I-FinvT*Finv
    % where Finv=I-dU
    ExxAll=(2*UXx-(UXx.^2+UYx.^2))/2;
    ExyAll=(UXy+UYx-(UXx.*UXy+UYx.*UYy))/2;
    EyxAll=ExyAll;
    EyyAll=(2*UYy-(UXy.^2+UYy.^2))/2;
    
    % Inplane principal strains Ep1All  and Ep2All are the eigenvalues of E
    % https://www.continuummechanics.org/principalstressesandstrains.html
    ThetaEp=0.5*atan(2*(ExyAll+EyxAll)/2./(ExxAll-EyyAll));%InPlanePrincipalOrientation
    Ep1All=(ExxAll+EyyAll)/2+sqrt(((ExxAll-EyyAll)/2).^2+((ExyAll+EyxAll)/2).^2);
    Ep2All=(ExxAll+EyyAll)/2-sqrt(((ExxAll-EyyAll)/2).^2+((ExyAll+EyxAll)/2).^2);
    % ThruPlane principal strain is calculated using the incompressibility
    % principle.
    Ep3All=Ep1All;
    Ep1All=max(Ep1All,Ep2All);
    Ep2All=min(Ep2All,Ep3All);
    Ep3All=1./(1+Ep1All)./(1+Ep2All)-1;
    
    % Here UX and UY are adjusted to guarantee that the principal strains caused by 
    % the simulated displacement, 
    % will not exceed the theshold StrainEp1Peak in each frame.  
    OverStrain=0;
    for iframe = 1:CineNoFrames
        MaxEp=max([max(squeeze(abs(Ep1All(:,:,iframe))),[],'all') max(squeeze(abs(Ep2All(:,:,iframe))),[],'all') max(squeeze(abs(Ep3All(:,:,iframe))),[],'all')]);
        %if abs(MaxEp-StrainEpPeak)>0.01
        if StrainEpPeak>MaxEp
            FrameDisplX(:,:,iframe)=(FrameDisplX(:,:,iframe))*max(0.95,StrainEpPeak/MaxEp);
            FrameDisplY(:,:,iframe)=(FrameDisplY(:,:,iframe))*max(0.95,StrainEpPeak/MaxEp);
            %FrameDisplX(:,:,iframe)=(FrameDisplX(:,:,iframe))*StrainEpPeak/MaxEp;
            %FrameDisplY(:,:,iframe)=(FrameDisplY(:,:,iframe))*StrainEpPeak/MaxEp;
            OverStrain=1;
        end
    end
end