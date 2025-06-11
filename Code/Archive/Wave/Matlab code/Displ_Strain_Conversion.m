    deltaT=1;

OverStrain=1;

while OverStrain==1
    UX=FrameDisplX; %UX The x-displacement of the material point at this specific spacial point 
    UY=FrameDisplY; %UY The y-displacement of the material point at this specific spacial point 
    
    %DisplxTotal=100*imEp1*deltaX;%smooth3(DisplxTotal,'gaussian',25,2);
    %DisplyTotal=100*imEp2*deltaY;%smooth3(DisplyTotal,'gaussian',25,2);
    
    [UXx, UXy, ExtAll]=gradient(UX,deltaX,deltaY,deltaT);
    [UYx, UYy, EytAll]=gradient(UY,deltaX,deltaY,deltaT);
    
    % Eulerian Strain Calculation based on the the left Cauchy-Green
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
        if MaxEp>StrainEpPeak
            FrameDisplX(:,:,iframe)=(FrameDisplX(:,:,iframe))*max(0.95,StrainEpPeak/MaxEp);
            FrameDisplY(:,:,iframe)=(FrameDisplY(:,:,iframe))*max(0.95,StrainEpPeak/MaxEp);
            OverStrain=1;
        end
    end

end

MaxDispl=max(max(UX(:)),max(UY(:)));
MinDispl=min(min(UX(:)),min(UY(:)));
