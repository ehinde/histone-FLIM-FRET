function PLICS_script()

%% Parameters
Param=struct;
FLAG =struct;

Param.PixelSize       =62;
Param.PixelUnits      ='nm';
Param.AreaUnits       ='\mum^2';
Param.TimeVect        =[-10 0 10 20 30 40 50 60 80 100 120 150 180 210 240 270 300 330 360];
Param.L_time          =length(Param.TimeVect);
Param.TimeUnits       ='min';

Param.PLICS_mask      =12;
Param.iPLICS_scale    =1/4;
Param.PLICS_mask_dist =36;
Param.Threshold       =0.1;
Param.CorrRadius      =7;
Param.Size_Gauss      =7;

Param.Range_Size      =[0 10]*Param.PixelSize;
Param.Range_Dist      =[400 1000];
Param.Range_G1        =[0 3];
Param.Range_N         =[0 3];

FLAG.LOAD             =1;
FLAG.OVERWRITE        =0;
FLAG.RECALCULATE_PLICS=0;
FLAG.DISPLAY          =1;
FLAG.FORCE_ANALYSIS   =[1 1 1 1];
FLAG.CHECK            =0;
FLAG.MASK_IMAGES      =1;

%% Load Data
if FLAG.LOAD==1
    fprintf('\n')
    fprintf(strcat('Loading Files...'))
    fprintf('\n')

%    EXP_name{1}='WT';
%    EXP_name{2}='ATM';
%    EXP_name{3}='RNF8';
    EXP_name{1}='Control';
    EXP_name{2}='No DSB';
    N_exp=length(EXP_name);

    EXP=cell(1,N_exp);
    EXP_filename=cell(1,N_exp);
    EXP_pathname=cell(1,N_exp);
    EXP_images=cell(1,N_exp);
    for i=1:N_exp
        fprintf(strcat('Select Data for experiment...',EXP_name{i}))
        fprintf('\n')
        [Images,FileName,PathName]=Open_Z64();
        EXP_filename{i}=FileName;
        EXP_pathname{i}=PathName;
        EXP_images{i}=Images;
    end        
end

%% Analysis - Experiment
for n_exp=1:N_exp
    
FileName=EXP_filename{n_exp};
PathName=EXP_pathname{n_exp};
Images=EXP_images{n_exp};

% Check if stack has the correct number of images
for flag_check=1
if FLAG.CHECK==1
    fprintf('\n')
    fprintf(strcat('Checking Files...'))
    fprintf('\n')
    L_images=length(Images);
    logic=true(1,L_images);
   for i=1:L_images
       if size(Images{i},3)==Param.L_time
       else
           logic(i)=0;
       end
   end    
   Images=Images(logic);
end
end
% Declare cells
for Declare_cells=1
L_images=length(Images);

SIZE=cell(1,L_images);
DIST=cell(1,L_images);
DISTmin=cell(1,L_images);
DISTmax=cell(1,L_images);
G1=cell(1,L_images);
N_map=cell(1,L_images);
end
for overwrite_load_experiment=1
    if FLAG.OVERWRITE==1
    FLAG_tmp=FLAG;
    Param_tmp=Param;
    load(strcat(EXP_pathname{n_exp},'Data_ALL.mat'));
    FLAG=FLAG_tmp;
    Param=Param_tmp;
    end
end

%% Analysis - Image Stack
for n_img=1:L_images

for select_data_path=1
A=Images{n_img};
A(A>0)=1;
Z=size(A,3);
SavePath=[PathName,FileName{n_img}(1:end-4),'\'];
cd 'C:\Users\Lorenzo\Documents\MATLAB'
clear BW
end    

%% Check if Image Stack has been already analyzed, if that's the case, load the data
for check_saved_data=1
if exist(strcat(SavePath,'Data.mat'),'file')==2&&FLAG.FORCE_ANALYSIS(n_exp)==0

% Load experiment
for overwrite_load_single_exp=1 
    fprintf('\n')
    fprintf(strcat('Analyzing Experiment...',num2str_dig(n_exp,2),'/',num2str_dig(N_exp,2),'_',EXP_name{n_exp}))
    fprintf('\n')
    
    if FLAG.OVERWRITE==1
    fprintf(strcat('Overwriting File.......',num2str_dig(n_img,2),'/',num2str_dig(L_images,2),'_',FileName{n_img}))
    else
    fprintf(strcat('Skipping File..........',num2str_dig(n_img,2),'/',num2str_dig(L_images,2),'_',FileName{n_img}))
    end
    fprintf('\n')
    fprintf(strcat('Loading From Previous Data - Substituting Parameters and Flags'))
    fprintf('\n')

    FLAG_tmp=FLAG;
    Param_tmp=Param;
    load(strcat(SavePath,'Data.mat'));
    FLAG=FLAG_tmp;
    Param=Param_tmp;
    
    if n_img==1
    size_curve=Size_curve;
    n_curve=N_curve;
    dist_curve=Dist_curve;
    D_global=Dist_Global;
    N_global=n_global;
    else
    size_curve(n_img,:)=Size_curve(1,:);
    n_curve(n_img,:)=N_curve(1,:);
    dist_curve(n_img,:)=Dist_curve(1,:);
    D_global(n_img,:)=Dist_Global(1,:);
    N_global(n_img,:)=n_global(1,:);
    end
end
% Overwrite Z
for overwrite_save_images=1
if FLAG.OVERWRITE==1    
    
fprintf(strcat('Overwriting Images...',FileName{n_img}))
fprintf('\n')

%% Save - PLICS size maps
if 1
for i=1:min(Z,Param.L_time)

    figure
    Col_bar(Size(:,:,i),Param.Range_Size,map,['t=',num2str_dig(Param.TimeVect(i),3)],Param.PixelUnits)
    saveas(gcf,strcat(SavePath,'Size_frame',num2str_dig(i,3)),'jpg');
    saveas(gcf,strcat(SavePath,'Size_frame',num2str_dig(i,3)),'fig');
    close
end
%% Save - iPLICS distance maps
for i=1:min(Z,Param.L_time)
    figure
    Col_bar(Dist(:,:,i),[Param.Range_Dist],map1,['t=',num2str_dig(Param.TimeVect(i),3)],Param.PixelUnits)
    saveas(gcf,strcat(SavePath,'Distance_frame',num2str_dig(i,3)),'jpg');
    saveas(gcf,strcat(SavePath,'Distance_frame',num2str_dig(i,3)),'fig');
    close
end
%% Save - PLICS G(1) maps
for i=1:min(Z,Param.L_time)
    figure
    Col_bar(g1(:,:,i),Param.Range_G1,map,['t=',num2str_dig(Param.TimeVect(i),3)],'G(1)')
    saveas(gcf,strcat(SavePath,'G1_frame',num2str_dig(i,3)),'jpg');
    saveas(gcf,strcat(SavePath,'G1_frame',num2str_dig(i,3)),'fig');
    close
end
end
%% Save - Sigle experiment parameters
Area=squeeze(sum(sum(BW)))'.*Param.PixelSize^2/10^6;
if Z<Param.L_time
    Area(end+1:Param.L_time)=0;
end
if Z>Param.L_time
    Area=Area(1:Param.L_time);
end
if n_img==1
Area_all=Area;
end
save(strcat(SavePath,'Data.mat'),'Size','Dist','g1','n_map','n_global','Size_curve','N_curve','Dist_curve','Dist_Global','Param','FLAG','BW','Area');
end 
end
for overwrite_stack=1
if FLAG.OVERWRITE==1
        SIZE{n_img}=Size;
        DIST{n_img}=Dist;
        G1{n_img}=g1;
        N_map{n_img}=n_map;
        N_global(n_img,:)=n_global;
        Area_all(n_img,:)=Area;
if Z<Param.L_time
        SIZE{n_img}(:,:,end+1:Param.L_time)=0;
        DIST{n_img}(:,:,end+1:Param.L_time)=0;
        G1{n_img}(:,:,end+1:Param.L_time)=0;
        N_map{n_img}(:,:,end+1:Param.L_time)=0;
        N_global(n_img,end+1:Param.L_time)=0;
        Area_all(n_img,end+1:Param.L_time)=0;
end
if Z>Param.L_time
        SIZE{n_img}=SIZE{n_img}(:,:,1:Param.L_time);
        DIST{n_img}=DIST{n_img}(:,:,1:Param.L_time);
        G1{n_img}=G1{n_img}(:,:,1:Param.L_time);
        N_map{n_img}=N_map{n_img}(:,:,1:Param.L_time);
        N_global=N_global(:,1:Param.L_time);
        Area_all=Area_all(:,1:Param.L_time);
end
 
end
end

else
    
for load_mask_or_determine=1
if exist(strcat(SavePath,'Data.mat'),'file')~=2    
mkdir(SavePath);
else
    variableInfo = who('-file', (strcat(SavePath,'Data.mat')));
    if ismember('BW', variableInfo)
        load(strcat(SavePath,'Data.mat'),'BW');
    end
end

fprintf('\n')
fprintf(strcat('Analyzing Experiment...',num2str_dig(n_exp,2),'/',num2str_dig(N_exp,2),'_',EXP_name{n_exp}))
fprintf('\n')
fprintf(strcat('Analyzing File.........',num2str_dig(n_img,2),'/',num2str_dig(L_images,2),'_',FileName{n_img}))
fprintf('\n')
end
% masking Z
for mask_determination=1
if FLAG.MASK_IMAGES==1&&exist('BW','var')==0
    check_mask=1;
    v=1:Z;
    while check_mask==1
for i=v
    mask_logic=MovGaussAverage_sigma(A(:,:,i),Param.Size_Gauss*3,Param.Size_Gauss);
    mask_logic(mask_logic>0)=1;
    A_dist_tmp=abs(A(:,:,i)-mask_logic*absmax(A(:,:,i)));
 %    
    if i==1&&length(v)==Z
    A_dist=A_dist_tmp;
    figure
    imagesc(A_dist(:,:,i));
    title(['select ROI for img:',num2str(i)])
    Figure_Format
    tmp = imfreehand;
    BW=tmp.createMask;
    close
    else
    A_dist(:,:,i)=A_dist_tmp;
    figure
    imagesc(A_dist(:,:,i));
    title(['select ROI for img:',num2str(i)])
    Figure_Format
    tmp = imfreehand;
    BW(:,:,i)=tmp.createMask;
    close
    end
    
end

figure
n_sub=ceil(sqrt(Z));
for i=1:Z
    subplot(n_sub,n_sub,i)
    imagesc(BW(:,:,i)+A_dist(:,:,i))
    Figure_Format
    title(num2str(i));
end
v=input('Redo any of the masks?');
close
if isempty(v)||sum(v==0)==1
    check_mask=0;
end
    end

else
    for i=1:Z
    mask_logic=MovGaussAverage_sigma(A(:,:,i),Param.Size_Gauss*3,Param.Size_Gauss);
    mask_logic(mask_logic>0)=1;
    A_dist_tmp=abs(A(:,:,i)-mask_logic*absmax(A(:,:,i)));
    if i==1
    A_dist=A_dist_tmp;
    else
    A_dist(:,:,i)=A_dist_tmp;
    end
    end

    if exist('BW','var')==0
    BW=ones(size(A));
    end
end
end

%% Analysis - Z
if FLAG.RECALCULATE_PLICS==1
for i=1:min(Z,Param.L_time)

    if i==1
        tic
    end
    ACF_store     =PLICS(A(:,:,i),Param.PLICS_mask,Param.Threshold,Param.PixelSize);
    ACF_store_dist =iPLICS(A_dist(:,:,i),Param.PLICS_mask_dist,Param.Threshold,Param.PixelSize,Param.iPLICS_scale,'Circle');

    if i==1
    T=toc;
    fprintf('\n')
    fprintf(strcat('Single Iteration Time=',num2str(round(T)),'seconds'))
    fprintf('\n')
    fprintf(strcat('Elapsed   Total  Time=',num2str(round(T*Z/60)),'minutes'))
    fprintf('\n')
    end
set(0,'DefaultFigureVisible','off')
    if i==1&&n_img==1
        SIZE{n_img}=ACF_store{1}(:,:,1).*BW(:,:,i);
        DIST{n_img}=ACF_store_dist{1}(:,:,1).*BW(:,:,i);
        DISTmin{n_img}=ACF_store_dist{2}(:,:,1).*BW(:,:,i);
        DISTmax{n_img}=ACF_store_dist{3}(:,:,1).*BW(:,:,i);
        G1{n_img}=ACF_store{4}(:,:,1).*BW(:,:,i);
        N_map{n_img}=ACF_store{5};
        N_global=ACF2D_N(A(:,:,i),Param.CorrRadius,0,1,0);
        Area_all=sum(sum(BW(:,:,i))).*Param.PixelSize^2/10^6;
        saveas(gcf,strcat(SavePath,'N_global',num2str_dig(i,3)),'jpg');
        saveas(gcf,strcat(SavePath,'N_global',num2str_dig(i,3)),'fig');
        close
        [a,b,c,D_global]=ACF2D_N(A_dist(:,:,i),Param.CorrRadius,0,1,0);
        saveas(gcf,strcat(SavePath,'Dist_global',num2str_dig(i,3)),'jpg');
        saveas(gcf,strcat(SavePath,'Dist_global',num2str_dig(i,3)),'fig');
        close
    else
        SIZE{n_img}(:,:,i)=ACF_store{1}(:,:,1).*BW(:,:,i);
        DIST{n_img}(:,:,i)=ACF_store_dist{1}(:,:,1).*BW(:,:,i);
        DISTmin{n_img}(:,:,i)=ACF_store_dist{2}(:,:,1).*BW(:,:,i);
        DISTmax{n_img}(:,:,i)=ACF_store_dist{3}(:,:,1).*BW(:,:,i);
        G1{n_img}(:,:,i)=ACF_store{4}(:,:,1).*BW(:,:,i);
        N_map{n_img}(:,:,i)=ACF_store{5};
        N_global(n_img,i)=ACF2D_N(A(:,:,i),Param.CorrRadius,0,1,0);
        Area_all(n_img,i)=sum(sum(BW(:,:,i))).*Param.PixelSize^2/10^6;
        saveas(gcf,strcat(SavePath,'N_global',num2str_dig(i,3)),'jpg');
        saveas(gcf,strcat(SavePath,'N_global',num2str_dig(i,3)),'fig');
        close
        [a,b,c,D_global(n_img,i)]=ACF2D_N(A_dist(:,:,i),Param.CorrRadius,0,1,0);
        saveas(gcf,strcat(SavePath,'Dist_global',num2str_dig(i,3)),'jpg');
        saveas(gcf,strcat(SavePath,'Dist_global',num2str_dig(i,3)),'fig');
        close
    end
set(0,'DefaultFigureVisible','on')
    close
    fprintf(strcat(num2str(min(Z,Param.L_time)-i),'-'))

end
for same_length=1
Area=squeeze(sum(sum(BW)))'.*Param.PixelSize^2/10^6; 
if Z<Param.L_time
        SIZE{n_img}(:,:,end+1:Param.L_time)=0;
        DIST{n_img}(:,:,end+1:Param.L_time)=0;
        DISTmin{n_img}(:,:,end+1:Param.L_time)=0;
        DISTmax{n_img}(:,:,end+1:Param.L_time)=0;
        G1{n_img}(:,:,end+1:Param.L_time)=0;
        N_map{n_img}(:,:,end+1:Param.L_time)=0;
        N_global(n_img,end+1:Param.L_time)=0;
        Area_all(n_img,end+1:Param.L_time)=0;
end
if Z>Param.L_time
        SIZE{n_img}=SIZE{n_img}(:,:,1:Param.L_time);
        DIST{n_img}=DIST{n_img}(:,:,1:Param.L_time);
        DISTmin{n_img}=DISTmin{n_img}(:,:,1:Param.L_time);
        DISTmax{n_img}=DISTmax{n_img}(:,:,1:Param.L_time);
        G1{n_img}=G1{n_img}(:,:,1:Param.L_time);
        N_map{n_img}=N_map{n_img}(:,:,1:Param.L_time);
        N_global=N_global(:,1:Param.L_time);
        Area_all=Area_all(:,1:Param.L_time);
end
if FLAG.DISPLAY==1
    set(0,'DefaultFigureVisible','on')
else
    set(0,'DefaultFigureVisible','off')
end

DIST{n_img}(DIST{n_img}<2)=0;

end

else
    load(strcat(SavePath,'Data.mat'));
end
%% Constructs average size, number and distance curves - Single Acquisition
for single_acquisition_curves=1
for i=1:min(Z,Param.L_time)
if i==1&&n_img==1
    size_curve=median(nonzeros(SIZE{n_img}(:,:,i)));
    n_curve=1./median(nonzeros(G1{n_img}(:,:,i)));
    dist_curve=median(nonzeros(DIST{n_img}(:,:,i)));
else
    size_curve(n_img,i)=median(nonzeros(SIZE{n_img}(:,:,i)));
    n_curve(n_img,i)=1./median(nonzeros(G1{n_img}(:,:,i)));
    dist_curve(n_img,i)=median(nonzeros(DIST{n_img}(:,:,i)));
end
end
if Z<Param.L_time
    size_curve(n_img,end+1:Param.L_time)=0;
    n_curve(n_img,end+1:Param.L_time)=0;
    dist_curve(n_img,end+1:Param.L_time)=0;
end

figure
yyaxis left
hold on
plot(Param.TimeVect,size_curve(n_img,:),'-ob')
%plot(Param.TimeVect,D_global(n_img,:)*Param.PixelSize,':*b')
ylabel(['Size (',Param.PixelUnits,')']);
xlabel(['Time (',Param.TimeUnits,')']);
yyaxis right
plot(Param.TimeVect,dist_curve(n_img,:),':*r')
ylabel(['Distance (',Param.PixelUnits,')']);
xlim([Param.TimeVect(1) Param.TimeVect(end)])
Figure_Format_Graph
saveas(gcf,strcat(SavePath,'Curves - Size and Distance'),'jpg');
saveas(gcf,strcat(SavePath,'Curves - Size and Distance'),'fig');

figure
plot(Param.TimeVect,N_global(n_img,:),'-o')
ylabel('N_G_l_o_b_a_l');
Figure_Format_Graph
title('Average Values over time')
xlim([Param.TimeVect(1) Param.TimeVect(end)])
saveas(gcf,strcat(SavePath,'Curves - Number'),'jpg');
saveas(gcf,strcat(SavePath,'Curves - Number'),'fig');

fprintf(strcat('Saving...',FileName{n_img}))
fprintf('\n')

%% Save - PLICS size maps
for i=1:min(Z,Param.L_time)
    figure
    Col_bar(SIZE{n_img}(:,:,i),Param.Range_Size,map,['t=',num2str_dig(Param.TimeVect(i),3)],Param.PixelUnits)
    saveas(gcf,strcat(SavePath,'Size_frame',num2str_dig(i,3)),'jpg');
    saveas(gcf,strcat(SavePath,'Size_frame',num2str_dig(i,3)),'fig');
    close
end
%% Save - iPLICS distance maps
for i=1:min(Z,Param.L_time)
    figure
    Col_bar(DIST{n_img}(:,:,i),[Param.Range_Dist],map1,['t=',num2str_dig(Param.TimeVect(i),3)],Param.PixelUnits)
    saveas(gcf,strcat(SavePath,'Distance_frame',num2str_dig(i,3)),'jpg');
    saveas(gcf,strcat(SavePath,'Distance_frame',num2str_dig(i,3)),'fig');
    close
end
%% Save - PLICS G(1) maps
for i=1:min(Z,Param.L_time)
    figure
    Col_bar(G1{n_img}(:,:,i),Param.Range_G1,map,['t=',num2str_dig(Param.TimeVect(i),3)],'G(1)')
    saveas(gcf,strcat(SavePath,'G1_frame',num2str_dig(i,3)),'jpg');
    saveas(gcf,strcat(SavePath,'G1_frame',num2str_dig(i,3)),'fig');
    close
end

%% Save - Sigle experiment parameters
Size=SIZE{n_img};
Dist=DIST{n_img};
g1=G1{n_img};
n_map=N_map{n_img};
n_global=N_global(n_img,:);
Dist_Global=D_global(n_img,:);
Area=squeeze(sum(sum(BW)))'.*Param.PixelSize^2/10^6;

for j=1:min(Z,Param.L_time)
    if j==1
Size_curve=median(nonzeros(Size(:,:,j)));
N_curve=1./median(nonzeros(g1(:,:,j)));    
Dist_curve=median(nonzeros(Dist(:,:,j)));
    else
Size_curve(j)=median(nonzeros(Size(:,:,j)));
N_curve(j)=1./median(nonzeros(g1(:,:,j)));    
Dist_curve(j)=median(nonzeros(Dist(:,:,j)));
    end
end
if Z<Param.L_time
    Size_curve(n_img,end+1:Param.L_time)=0;
    N_curve(n_img,end+1:Param.L_time)=0;
    Dist_curve(n_img,end+1:Param.L_time)=0;
end

save(strcat(SavePath,'Data.mat'),'Size','Dist','g1','n_map','n_global','Size_curve','N_curve','Dist_curve','Dist_Global','Param','FLAG','BW','Area');

end
end
end

end

%% Constructs average size, number and distance curves - Whole Experiment
for curves_whole_experiment=1

size_curve_mean=median_nonzeros(size_curve,1);
size_curve_std=std_nonzeros(size_curve,[],1);
dist_curve_mean=median_nonzeros(dist_curve,1);
dist_curve_std=std_nonzeros(dist_curve,[],1);
D_global_mean=median_nonzeros(D_global,1);
D_global_std=std_nonzeros(D_global,[],1);
tmp=N_global./Area_all;
tmp(isnan(tmp))=0;
tmp(isinf(tmp))=0;
Density_global=tmp;
Density_global_mean=median_nonzeros(tmp,1);
Density_global_std=std_nonzeros(tmp,[],1);

figure
yyaxis left
hold on
errorbar(Param.TimeVect,size_curve_mean,size_curve_std,'-b')
%errorbar(Param.TimeVect,D_global_mean*Param.PixelSize,D_global_std*Param.PixelSize,'-b')
plot(Param.TimeVect,size_curve_mean,'-b','LineWidth',2,'MarkerSize',8,'Marker','o')
%plot(Param.TimeVect,D_global_mean*Param.PixelSize,'-*b','LineWidth',2,'MarkerSize',10)
ylabel(['Size (',Param.PixelUnits,')']);
xlabel(['Time (',Param.TimeUnits,')']);
xlim([Param.TimeVect(1) Param.TimeVect(end)])
Figure_Format_Graph
yyaxis right
ylabel(['Distance (',Param.PixelUnits,')']);
errorbar(Param.TimeVect,dist_curve_mean,dist_curve_std,'-r')
plot(Param.TimeVect,dist_curve_mean,'-r','LineWidth',2,'MarkerSize',10,'Marker','*')
saveas(gcf,strcat(EXP_pathname{n_exp},'AverageValues - Size and Distance'),'fig');
saveas(gcf,strcat(EXP_pathname{n_exp},'AverageValues - Size and Distance'),'jpg');

figure
errorbar(Param.TimeVect,Density_global_mean,Density_global_std,'-r')
plot(Param.TimeVect,Density_global_mean,'-','LineWidth',2,'MarkerSize',8,'Marker','o')
ylabel(['N_d_e_n_s_i_t_y (#/',Param.AreaUnits,')']);
xlabel(['Time (',Param.TimeUnits,')']);
Figure_Format_Graph
title('Density over time')
xlim([Param.TimeVect(1) Param.TimeVect(end)])
ylim([0 Inf])
saveas(gcf,strcat(EXP_pathname{n_exp},'AverageValues - Number'),'fig');
saveas(gcf,strcat(EXP_pathname{n_exp},'AverageValues - Number'),'jpg');

%% Save - Whole Experiment Data
save(strcat(EXP_pathname{n_exp},'Data_ALL.mat'),'SIZE','DIST','G1','N_map','Density_global','Density_global_mean','Density_global_std','N_global','size_curve','size_curve_mean','size_curve_std','D_global','D_global_mean','D_global_std','n_curve','dist_curve','Param','FLAG','Area_all');
end
end

%% Compare experiments
for CompareExperiments=1
    
    titles_cell_sheet1=cell(1);
    variables_cell_sheet1=cell(1);
    titles_cell_sheet2=cell(1);
    variables_cell_sheet2=cell(1);
    titles_cell_sheet3=cell(1);
    variables_cell_sheet3=cell(1);
    
    titles_cell_sheet1{1}='Time Vector';
    variables_cell_sheet1{1}=Param.TimeVect;
    titles_cell_sheet2{1}='Time Vector';
    variables_cell_sheet2{1}=Param.TimeVect;
    titles_cell_sheet3{1}='Time Vector';
    variables_cell_sheet3{1}=Param.TimeVect;
    
fig_size=figure;
fig_D=figure;
fig_N_global=figure;
fig_size_short=figure;
fig_D_short=figure;
fig_N_global_short=figure;
short_n=(1:8);

co=get(groot,'defaultAxesColorOrder');

for n_exp=1:N_exp   
load(strcat(EXP_pathname{n_exp},'Data_ALL.mat'));

dist_curve_mean=median_nonzeros(dist_curve,1);
dist_curve_std=std_nonzeros(dist_curve,[],1);

figure(fig_size)
hold on
errorbar(Param.TimeVect,size_curve_mean,size_curve_std,'-o','Color',co(n_exp,:))
plot(Param.TimeVect,size_curve_mean,'-','Color',co(n_exp,:),'MarkerSize',8,'Marker','o')
figure(fig_D)

titles_cell_sheet1{2+(n_exp-1)*2}='Size (mean)';
variables_cell_sheet1{2+(n_exp-1)*2}=size_curve_mean;
titles_cell_sheet1{3+(n_exp-1)*2}='Size (std)';
variables_cell_sheet1{3+(n_exp-1)*2}=size_curve_std;

hold on
errorbar(Param.TimeVect,dist_curve_mean,dist_curve_std,'-o','Color',co(n_exp,:))
plot(Param.TimeVect,dist_curve_mean,'-','Color',co(n_exp,:),'MarkerSize',8,'Marker','o')

titles_cell_sheet2{2+(n_exp-1)*2}='Distance (mean)';
variables_cell_sheet2{2+(n_exp-1)*2}=dist_curve_mean;
titles_cell_sheet2{3+(n_exp-1)*2}='Distance (std)';
variables_cell_sheet2{3+(n_exp-1)*2}=dist_curve_std;

figure(fig_N_global)
hold on
errorbar(Param.TimeVect,Density_global_mean,Density_global_std,'-','Color',co(n_exp,:))
plot(Param.TimeVect,Density_global_mean,'-o','Color',co(n_exp,:),'MarkerSize',8,'Marker','o')

titles_cell_sheet3{2+(n_exp-1)*2}='Density (mean)';
variables_cell_sheet3{2+(n_exp-1)*2}=Density_global_mean;
titles_cell_sheet3{3+(n_exp-1)*2}='Density (std)';
variables_cell_sheet3{3+(n_exp-1)*2}=Density_global_std;

figure(fig_size_short)
hold on
errorbar(Param.TimeVect(short_n),size_curve_mean(short_n),size_curve_std(short_n),'-o','Color',co(n_exp,:))
plot(Param.TimeVect(short_n),size_curve_mean(short_n),'-','Color',co(n_exp,:),'MarkerSize',8,'Marker','o')
figure(fig_D_short)
hold on
errorbar(Param.TimeVect(short_n),dist_curve_mean(short_n),dist_curve_std(short_n),'-o','Color',co(n_exp,:))
plot(Param.TimeVect(short_n),dist_curve_mean(short_n),'-','Color',co(n_exp,:),'MarkerSize',8,'Marker','o')
figure(fig_N_global_short)
hold on
errorbar(Param.TimeVect(short_n),Density_global_mean(short_n),Density_global_std(short_n),'-','Color',co(n_exp,:))
plot(Param.TimeVect(short_n),Density_global_mean(short_n),'-o','Color',co(n_exp,:),'MarkerSize',8,'Marker','o')


end

figure(fig_size)
ylabel(['Size (',Param.PixelUnits,')']);
xlabel(['Time (',Param.TimeUnits,')']);
xlim([Param.TimeVect(1) Param.TimeVect(end)])
Figure_Format_Graph
title('Size VS Time')
saveas(gcf,'Size VS Time','fig');
saveas(gcf,'Size VS Time','jpg');
figure(fig_size_short)
ylabel(['Size (',Param.PixelUnits,')']);
xlabel(['Time (',Param.TimeUnits,')']);
xlim([Param.TimeVect(short_n(1)) Param.TimeVect(short_n(end))])
Figure_Format_Graph
title('Size VS Time')
saveas(gcf,'Size VS Time - Short','fig');
saveas(gcf,'Size VS Time - Short','jpg');

figure(fig_D)
ylabel(['Distance (',Param.PixelUnits,')']);
xlabel(['Time (',Param.TimeUnits,')']);
xlim([Param.TimeVect(1) Param.TimeVect(end)])
Figure_Format_Graph
title('Distance VS Time')
ylim([0 Inf])
saveas(gcf,'Distance VS Time','fig');
saveas(gcf,'Distance VS Time','jpg');
figure(fig_D_short)
ylabel(['Distance (',Param.PixelUnits,')']);
xlabel(['Time (',Param.TimeUnits,')']);
xlim([Param.TimeVect(short_n(1)) Param.TimeVect(short_n(end))])
Figure_Format_Graph
title('Distance VS Time')
%ylim([0 Inf])
saveas(gcf,'Distance VS Time - Short','fig');
saveas(gcf,'Distance VS Time - Short','jpg');

figure(fig_N_global)
ylabel('N_G_l_o_b_a_l');
ylabel(['N_d_e_n_s_i_t_y (#/',Param.AreaUnits,')']);
xlabel(['Time (',Param.TimeUnits,')']);
Figure_Format_Graph
title('Density over time')
ylim([0 Inf])
saveas(gcf,'Number VS Time','fig');
saveas(gcf,'Number VS Time','jpg');
figure(fig_N_global_short)
ylabel(['N_d_e_n_s_i_t_y (#/',Param.AreaUnits,')']);
xlabel(['Time (',Param.TimeUnits,')']);
Figure_Format_Graph
title('Density over time')
xlim([Param.TimeVect(short_n(1)) Param.TimeVect(short_n(end))])
%ylim([0 Inf])
saveas(gcf,'Number VS Time - Short','fig');
saveas(gcf,'Number VS Time - Short','jpg');
end

xlswrite_custom('Data Cells - Comparison',titles_cell_sheet1,variables_cell_sheet1,1)
xlswrite_custom('Data Cells - Comparison',titles_cell_sheet2,variables_cell_sheet2,2)
xlswrite_custom('Data Cells - Comparison',titles_cell_sheet3,variables_cell_sheet3,3)

end

%% Other functions

function [N,ACF,ACF1,FWHM]=ACF2D_N(A,CorrelationRadius,threshold,Display,comp2)

if nargin<5
    comp2=0;
if nargin<4
    Display=0;
end
end
if threshold==0
else
A=A-threshold;
A(A<0)=0;
end
%A(A<threshold)=0;
if sum(sum(A==0))==size(A,1)*size(A,2)

    N=0;
    ACF=0;
    ACF1=0;
    FWHM=0;
    
else
    
[X,Y]=size(A);
F=fft2(A);
ACF= F.*conj(F);
G=(sum(sum((A))))^2/X/Y;
ACF= ifft2(ACF);
ACF= fftshift(ACF)./G-1;
ACF1=ACF;

if iseven(size(A,1))
r=round(size(A,1)/2+1);
c=round(size(A,2)/2+1);
else
r=round(size(A,1)+1)/2;
c=round(size(A,2)+1)/2;
end    
ACF1=ACF1(r-CorrelationRadius:r+CorrelationRadius,c-CorrelationRadius:c+CorrelationRadius);
ACF=ACF(r-CorrelationRadius:r+CorrelationRadius,c-CorrelationRadius:c+CorrelationRadius);

[r,c]=find(ACF1==absmax(ACF1));
ACF1(r,c)=0;
ACF1(r,c)=absmax(ACF1);

if ACF1==0
    N=0;
else
    if comp2
[param,K1]=Fit2DGaussian_asym(ACF1,0);
[param,K]=Fit2DGaussian_asym(ACF1-K1,Display);
    else
[param,K]=Fit2DGaussian_asym(ACF1,Display);
    end
N=X*Y./(param(1)*2*pi*min(param(2:3))^2);
FWHM=min(param(2:3))*2.35;
end

end
end

function K=MovGaussAverage_sigma(B,MaskSize,sigma)
if sigma==0
    K=B;
else
[X,Y,Z]=size(B);

L=(MaskSize-1)/2;
B1=ones(X+2*L,Y+2*L,Z);
for i=1:Z
    B1(:,:,i)=B1(:,:,i)*mean2(B(:,:,i));
end
B1(L+1:end-L,L+1:end-L,:)=B;
% for i=1:Z
%    tmp=B1(:,:,i);
%    tmp(tmp==0)=mean2_0(tmp);
%    B1(:,:,i)=tmp;
% end

K=zeros(X+2*L,Y+2*L,Z);
g=Gaussian2D(MaskSize,sigma);
g=g/sum(sum(g));

for i=1:Z
K(:,:,i)=conv2(B1(:,:,i),g,'same');
end

K=K(L+1:end-L,L+1:end-L,:);
end
end
function g2=Gaussian2D(DimMatrix,sigma)

g2=zeros(DimMatrix);
x=(1:1:DimMatrix);
g=Gaussian(x,sigma,(DimMatrix+1)/2,1);

for i=1:1:DimMatrix
    g2(i,1:end)=g*g(i);
end
end
function g=Gaussian(x,sigma,center,height)

%g=height*(1/(sqrt(2*sigma.^2)))*(exp(-(x-center).^2/(2*sigma.^2)));
g=height*(exp(-(x-center).*(x-center)/(2*sigma.*sigma)));

end
function a=absmax(A)

SIZE=ndims(A)-sum(size(A)==1);

switch SIZE
    case 0
        a=A;
    case 1
        a=max(A);
    case 2
        a=max(max(A));
    case 3
        a=max(max(max(A)));
    case 4
        a=max(max(max(max(A))));
end
end
function a=absmin(A)

[r,s,t]=size(A);
SIZE=3;

if t==1
    SIZE=2;
    if r==1||s==1
        SIZE=1;
    end
end

switch SIZE
    case 1
        a=min(A);
    case 2
        a=min(min(A));
    case 3
        a=min(min(min(A)));
end
end
function bool=iseven(x)

if mod(x,2) == 0
bool=1;
else
bool=0;
end
endfunction cmap = map
% Returns my custom colormap
cmap=colormap(jet);
cmap(1,:)=[0,0,0];
cmap(65,:)=[1,1,1];
end
function  [Output,OrientedProfiles,Angles]=ACFmean_test1_G1(x)

[X,Y] = size(x);
Oversampling=0;
NumberOfAngles=20;
min_th_row=zeros(1,NumberOfAngles*2);

% calculate 2D ACF by FFT 
F=fft2(x);
ACF= F.*conj(F);
G=((sum(sum(x)))^2/X/Y);
ACF= ifft2(ACF);
ACF= fftshift(ACF)./G-1;
%ACF(ACF==absmax(ACF))=0;

if Oversampling>1
ACF=Oversample_Int(ACF,Oversampling);
end


[R, C]=size(ACF);

r0=R/2+1;
c0=C/2+1;

ACF1=flipud(ACF(2:r0,c0:end));
ACF2=ACF(r0:end,c0:end);
Radius=min(R/2,C/2);
ProfMat=zeros(NumberOfAngles*2,Radius);

for j=1:2
    if j==1
        y=ACF1';
    else
        y=ACF2;
    end
    
% CALCULATION OF ROTATIONAL MEAN
% Definition of angles
t=(pi/NumberOfAngles/2:pi/NumberOfAngles/2:pi/2);
   
% Matrix
y=y(1:Radius,1:Radius);
% Cycle between the 2nd and 2nd to last angles
[~, y1y]=size(y);

for i=1:NumberOfAngles
   rt=ceil(cos(t(i))*(1:Radius));
   ct=ceil(sin(t(i))*(1:Radius));
   profile=y((rt-1).*y1y+ct);

   if j==1
   ProfMat(NumberOfAngles+i,:)=profile;
   else
   ProfMat(i,:)=profile;
   end   
end

end

fwhm_1=ProfMat-absmin(ProfMat);
fwhm_1=abs(fwhm_1./absmax(fwhm_1)-0.5);
min_fw=min(fwhm_1,[],2);

for k=1:NumberOfAngles*2
    min_th_row1=ceil(find(fwhm_1(k,:)==min_fw(k)));
    if isempty(min_th_row1)
        min_th_row(k)=0;
    else
        min_th_row(k)=min_th_row1(1);
    end
end

min_th_row=find(min_th_row==min(min_th_row));
min_th=min_th_row(1)*pi/2/NumberOfAngles;

if min_th_row(1)<=NumberOfAngles
    max_th=min_th+pi/2;
    max_th_row=min_th_row(1)+NumberOfAngles;
else
    max_th=min_th-pi/2;
    max_th_row=min_th_row(1)-NumberOfAngles;
end

Prof2Mean=10;
ProfMat1=zeros(2*NumberOfAngles+Prof2Mean*2,Radius);
ProfMat1(1:Prof2Mean,:)=ProfMat(end-Prof2Mean+1:end,:);
ProfMat1(1+Prof2Mean:NumberOfAngles*2+Prof2Mean,:)=ProfMat;
ProfMat1(NumberOfAngles*2+Prof2Mean+1:NumberOfAngles*2+Prof2Mean*2,:)=ProfMat(1:Prof2Mean,:);
min_th_row=min_th_row+Prof2Mean;
max_th_row=max_th_row+Prof2Mean;

if Oversampling>1
min_fw_Profile=DSample1D(mean(ProfMat1(min_th_row(1)-Prof2Mean:min_th_row(1)+Prof2Mean,:),1),Oversampling);
max_fw_Profile=DSample1D(mean(ProfMat1(max_th_row(1)-Prof2Mean:max_th_row(1)+Prof2Mean,:),1),Oversampling);
Output=DSample1D(sum(ProfMat)/(2*NumberOfAngles),Oversampling);
else
min_fw_Profile=sum(ProfMat1(min_th_row(1)-Prof2Mean:min_th_row(1)+Prof2Mean,:),1)./(2*Prof2Mean+1);
max_fw_Profile=sum(ProfMat1(max_th_row(1)-Prof2Mean:max_th_row(1)+Prof2Mean,:),1)./(2*Prof2Mean+1);
Output=sum(ProfMat)./(2*NumberOfAngles);
end

Output=Output(2:end);
OrientedProfiles=min_fw_Profile(2:end);
OrientedProfiles(2,:)=max_fw_Profile(2:end);
Angles=[min_th,max_th];

end
function Figure_Format()
FigHandle = gcf;
set(FigHandle, 'Position', [450, 10, 800, 800]);
%axis image
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'Ztick',[])
set(gca,'Zticklabel',[])
xlabel('')
ylabel('')
zlabel('')
%FName=get(gcf,'FileName');
%saveas(FigHandle,FName);
%saveas(FigHandle,strcat(FName(1:end-4),'.tif'));

%close all
end
function Figure_Format_Graph()

FigHandle = gcf;
set(FigHandle, 'Position', [450, 10, 800, 800]);

end
function y=DSample1D(x,mask)

y=zeros(1);

    for i=1:floor(length(x)/mask)
            
            y(i)=x(1+(i-1)*mask);

    end
    end
function [param, K]=Fit2DGaussian_asym(H,Display)
param0=[max(max(H)) 2 2 0 0 -1 mean2(H)];
[param, ~]=fminsearch(@(Param)sum(sum((Param(1).*Gaussian2D_asym(Param(5),Param(6),size(H,1),Param(2),Param(3),Param(4))+Param(7)-H).^2)),param0,optimset('Display','none'));
K=param(1).*Gaussian2D_asym(param(5),param(6),size(H,1),param(2),param(3),param(4))+param(7);

if Display==1
    figure
    subplot(2,2,1)
    imagesc(H)
    axis image
    title('Original Image')
    subplot(2,2,2)
    imagesc(K)
    axis image
    title('Best Guess')
    subplot(2,2,3)
    imagesc(H-K)
    colormap(jet)
    title('Residuals')
    subplot(2,2,4)
    surf(H-K)
    colormap(jet)
    title('Residuals')
    colormap(hot(256))
    Figure_Format_Graph
end
    
end
function g=Gaussian2D_asym(x0,y0,DimMatrix,sigma_x,sigma_y,th)

g=zeros(DimMatrix);
x=(1:1:DimMatrix);
y=(1:1:DimMatrix);

a=(cos(th)*cos(th))/(2*sigma_x*sigma_x)+(sin(th)*sin(th))/(2*sigma_y*sigma_y);
b=-sin(2*th)/(4*sigma_x*sigma_x)+sin(2*th)/(4*sigma_y*sigma_y);
c=(sin(th)*sin(th))/(2*sigma_x*sigma_x)+(cos(th)*cos(th))/(2*sigma_y*sigma_y);

for i=1:length(y)
    for j=1:length(x)
        a1=(x(j)-(DimMatrix+1-x0)/2);
        b1=(y(i)-(DimMatrix+1-y0)/2);
        g(i,j)=exp(-a1.*a1.*a+2*b*(x(j)-x0-(DimMatrix+1)/2)*(y(i)-y0-(DimMatrix+1)/2)-b1.*b1.*c);
    end
end
end
function Col_bar(x,K,Colormap,ImgTitle,BarTitle)

if nargin<4
    ImgTitle='';
    BarTitle='';
if nargin==1
    K=0;
    Colormap=map;
end
end

if length(K)==2
m=K(1);
M=K(2);
else
M=absmax(nonzeros(x));
m=absmin(nonzeros(x));
if isempty(m)
    m=0;
end
if isempty(M)
    M=0;
end
end

if m==M
    M=m+10^-5;
end
positionVector1 = [0.05, 00.05, 0.75, 0.9];
title(BarTitle)
subplot('Position',positionVector1)
imagesc(x,[m M])
title(ImgTitle)
colormap(Colormap)
axis image
Figure_Format
g=gcf;
v=g.Position;

height=200/v(3)*3;
y=201;
l=flip((m:(M-m)/(y-1):M));
col_bar=repmat(l',1,2);
positionVector2 = [0.81, (1-height)/2, 0.05, height];
subplot('Position',positionVector2)
imagesc(col_bar)
%axis image
set(gca,'YAxisLocation','right');
set(gca,'YTick',(1:round(y/10):y))
set(gca,'YTickLabel',flip(round((m:(M-m)/10:M),3)))
set(gca,'FontSize',21);
set(gca,'XTickLabel',[])
positionVector1 = [0.05, 00.05, 0.75, 0.9];
title(BarTitle)
subplot('Position',positionVector1)
imagesc(x,[m M])
title(ImgTitle)
colormap(Colormap)
axis image
Figure_Format

end
function A=mean2(B)

[r,c]=size(B);
A=sum(sum(B))./(r*c);

end
function [Images,FileName,PathName]=Open_Z64()

[FileName,PathName] = uigetfile('*.Z64','MultiSelect','On');
% read .Z64 binary file
if ischar(FileName)
    tmp=FileName;
    FileName=cell(1);
    FileName{1}=tmp;
end
L=length(FileName);
Images=cell(1,L);
for i=1:L
    
fileID = fopen([PathName FileName{i}]);
input = fread(fileID);
fclose(fileID);
 % decompress file content
 buffer = java.io.ByteArrayOutputStream();
 zlib = java.util.zip.InflaterOutputStream(buffer);
 zlib.write(input, 0, numel(input));
 zlib.close();
 buffer = buffer.toByteArray();
 % read image dimension, number of images, and image data from 
 % decompressed buffer
 sizes = typecast(buffer(1:4), 'int32');
 nimages = typecast(buffer(5:8), 'int32');
 images = reshape(typecast(buffer(9:end), 'single'), sizes, sizes, nimages);
 Images{i}=double(images);

end 
end
function B=Oversample_Int(A,OS)

[R,C]=size(A);
[X,Y] = meshgrid(linspace(1,C,C),linspace(1,R,R));
[X1,Y1] = meshgrid(linspace(1,C,C*OS),linspace(1,R,R*OS));
B=interp2(X,Y,A,X1,Y1,'cubic');
end

%% PLICS functions
function ACF_store=PLICS(x,masksize,threshold,PixelSize,Type)
%% INPUTS:
%$£ - x          : Image to be analyzed
%$£ - masksize   : Size of the mask (minimum 8x8, only even masks)
%$£ - threshold  : Threshold, defines the number of points to be analyzed
%$£ - PixelSize  : Pixel size, scales to a real value

%% OUTPUTS:
%$£ - ACF_store  : Cell array with the results of the analysis, the file are
%$£                arranged so that:
%$£    -{1}      : Rotational Average profile
%$£    -{2}      : Minimum Profile
%$£    -{3}      : Maximum Profile
%$£    -{4}      : G(1) map
%% 
%$£ For every location of the cell array results a 3D image, the image
%$£ (:,:,1) is the size image while (:,:,2) contains the values of (g,s) for
%$£ the phasor plot. All non-analyzed values are set to 0.
%$£
%%%%%%%%% Lorenzo Scipioni - IIT Nanophysics Department, Genova%%%%%%%%%%%%



if nargin<5
    Type='Gauss';
if nargin<4
    PixelSize=1;
if nargin<3
    threshold=0;
end
end
end

MaxPhasorOrder=1;
if strcmp(Type,'Gauss')
[C0,C1]=CalibrationParam_single_new01(masksize,MaxPhasorOrder);
end
if strcmp(Type,'Circle')
[C0,C1]=CalibrationParam_single_Circle01(masksize);
end

ACF_store=cell(1);

    x_thr=x;
    x_thr(x_thr<threshold)=0;
    x_thr(x_thr>0)=1;
    skip=5;
    x_thr(1:skip,:)=0;
    x_thr(end-skip:end,:)=0;
    x_thr(:,1:skip)=0;
    x_thr(:,end-skip:end)=0;

[m,n]=size(x);

tmp=zeros(m+masksize+2,n+masksize+2);
tmp(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1)=x_thr;
x_thr=tmp;

x1=zeros(m+masksize+2,n+masksize+2);
x1(1+masksize/2+1:m+masksize/2+1,1+masksize/2+1:n+masksize/2+1)=x;

Size_m=zeros(m+masksize+2,n+masksize+2);
Size_1=zeros(m+masksize+2,n+masksize+2);
Size_2=zeros(m+masksize+2,n+masksize+2);
GS_m=zeros(m+masksize+2,n+masksize+2);
GS_1=zeros(m+masksize+2,n+masksize+2);
GS_2=zeros(m+masksize+2,n+masksize+2);

G1_map=zeros(m+masksize+2,n+masksize+2);

%Even Mask

while sum(sum(x_thr==1))~=0
    
    x2=x1.*x_thr;
    pos=find(x2==absmax(x2));
    [i,j]=ind2sub([m+masksize+2,n+masksize+2],pos(1));
            
        A=x1(i-masksize/2:i+masksize/2-1,j-masksize/2:j+masksize/2-1);
        [RotProfile,OrientedProfiles,~]=ACFmean_test0_G1(A);
        
        %Insert calibration and size transformation here 
        [gf_m,Phase_m,~,Phase_S_m,~]=PhasorPlot_01_SingleProfile(RotProfile,MaxPhasorOrder);
        [gf_1,Phase_1,~,Phase_S_1,~]=PhasorPlot_01_SingleProfile(OrientedProfiles(1,:),MaxPhasorOrder);
        [gf_2,Phase_2,~,Phase_S_2,~]=PhasorPlot_01_SingleProfile(OrientedProfiles(2,:),MaxPhasorOrder);
        
        for k=1:MaxPhasorOrder
        %$£ The chosen value if the shifted phase value
        
                Phase_m(Phase_m<0)=0;
                Phase_1(Phase_1<0)=0;
                Phase_2(Phase_2<0)=0;
                
                Phase_S_m(Phase_S_m<0)=Phase_S_m(Phase_S_m<0)+pi;
                Phase_S_1(Phase_S_1<0)=Phase_S_1(Phase_S_1<0)+pi;
                Phase_S_2(Phase_S_2<0)=Phase_S_2(Phase_S_2<0)+pi;

                Phase_m=real((atanh((Phase_m-C0(1))./C0(2))./C0(3))+C0(4));
                Phase_1=real((atanh((Phase_1-C0(1))./C0(2))./C0(3))+C0(4));
                Phase_2=real((atanh((Phase_2-C0(1))./C0(2))./C0(3))+C0(4));
                Phase_S_m=real((atanh((Phase_S_m-C1(1))./C1(2))./C1(3))+C1(4));
                Phase_S_1=real((atanh((Phase_S_1-C1(1))./C1(2))./C1(3))+C1(4));
                Phase_S_2=real((atanh((Phase_S_2-C1(1))./C1(2))./C1(3))+C1(4));

        %$£$£ Checks and corrects for saturated values

                Phase_m(real(Phase_m)>masksize/k)=masksize/k;
                Phase_S_m(real(Phase_S_m)>masksize/k)=masksize/k;
                Phase_1(real(Phase_1)>masksize/k)=masksize/k;
                Phase_S_1(real(Phase_S_1)>masksize/k)=masksize/k;
                Phase_2(real(Phase_2)>masksize/k)=masksize/k;
                Phase_S_2(real(Phase_S_2)>masksize/k)=masksize/k;
                Phase_S_m(isnan(Phase_S_m))=0;
                              
                
        end
        
  %$£ Store size and phasor values      

  if imag(round(Phase_S_m*2+1))~=0
      g=0;
  end
  
  if iseven(round(Phase_S_m*2+1))
            
        Size_m(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)=Size_m(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)+Circle(round(Phase_S_m*2+1)+1).*Phase_S_m.*x_thr(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2);
        Size_1(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)=Size_1(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)+Circle(round(Phase_S_m*2+1)+1).*Phase_S_1.*x_thr(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2);
        Size_2(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)=Size_2(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)+Circle(round(Phase_S_m*2+1)+1).*Phase_S_2.*x_thr(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2);
        GS_m(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)=GS_m(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)+Circle(round(Phase_S_m*2+1)+1).*gf_m(1).*x_thr(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2);
        GS_1(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)=GS_1(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)+Circle(round(Phase_S_m*2+1)+1).*gf_1(1).*x_thr(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2);
        GS_2(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)=GS_2(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)+Circle(round(Phase_S_m*2+1)+1).*gf_2(1).*x_thr(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2);
        G1_map(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)=G1_map(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)+Circle(round(Phase_S_m*2+1)+1).*RotProfile(1).*x_thr(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2);
        
        x_thr(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)=x_thr(i-round(Phase_S_m*2+1)/2:i+round(Phase_S_m*2+1)/2,j-round(Phase_S_m*2+1)/2:j+round(Phase_S_m*2+1)/2)-Circle(round(Phase_S_m*2+1)+1);
        x_thr(x_thr<0)=0;
        else
            
        Size_m(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)=Size_m(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)+Circle(round(Phase_S_m*2)+1).*Phase_S_m.*x_thr(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2);
        Size_1(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)=Size_1(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)+Circle(round(Phase_S_m*2)+1).*Phase_S_1.*x_thr(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2);
        Size_2(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)=Size_2(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)+Circle(round(Phase_S_m*2)+1).*Phase_S_2.*x_thr(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2);
        GS_m(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)=GS_m(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)+Circle(round(Phase_S_m*2)+1).*gf_m(1).*x_thr(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2);
        GS_1(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)=GS_1(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)+Circle(round(Phase_S_m*2)+1).*gf_1(1).*x_thr(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2);
        GS_2(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)=GS_2(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)+Circle(round(Phase_S_m*2)+1).*gf_2(1).*x_thr(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2);
        G1_map(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)=G1_map(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)+Circle(round(Phase_S_m*2)+1).*RotProfile(1).*x_thr(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2);

        x_thr(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)=x_thr(i-round(Phase_S_m*2)/2:i+round(Phase_S_m*2)/2,j-round(Phase_S_m*2)/2:j+round(Phase_S_m*2)/2)-Circle(round(Phase_S_m*2)+1);
        x_thr(x_thr<0)=0;
  end
  
  
  
end
    
GS_m=conj(GS_m);
GS_m(isnan(GS_m))=0;
GS_1=conj(GS_1);
GS_1(isnan(GS_1))=0;
GS_2=conj(GS_2);
GS_2(isnan(GS_2))=0;

ACF_store{1}=real(Size_m(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1))*PixelSize;
ACF_store{1}(:,:,2)=real(GS_m(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1));
ACF_store{2}=real(Size_1(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1))*PixelSize;
ACF_store{2}(:,:,2)=real(GS_1(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1));
ACF_store{3}=real(Size_2(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1))*PixelSize;
ACF_store{3}(:,:,2)=real(GS_2(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1));
ACF_store{4}=G1_map(masksize/2+2:end-masksize/2-1,masksize/2+2:end-masksize/2-1);
ACF_store{5}=(masksize*PixelSize)^2/pi./(ACF_store{1}(:,:,1).^2)./ACF_store{4};
ACF_store{5}(isinf(ACF_store{5}))=0;
ACF_store{5}(isnan(ACF_store{5}))=0;
end
function ACF_store=iPLICS(x,masksize,threshold,PixelSize,Scale_reassign,Type,ACF_store_PLICS)

if nargin<6
    Type='Gauss';
if nargin<5
    Scale_reassign=1;
if nargin<4
    PixelSize=1;
if nargin<3
    threshold=0;
end
end
end
end

MaxPhasorOrder=1;
if strcmp(Type,'Gauss')
[C0,C1]=CalibrationParam_single_new01(masksize,MaxPhasorOrder);
end
if strcmp(Type,'Circle')
[C0,C1]=CalibrationParam_single_Circle01(masksize);
end
ACF_store=cell(1);

    x_thr=x;
    x_thr(x_thr<threshold)=0;
    x_thr(x_thr>0)=1;
    skip=masksize/2;
    x_thr(1:skip,:)=0;
    x_thr(end-skip:end,:)=0;
    x_thr(:,1:skip)=0;
    x_thr(:,end-skip:end)=0;

[m,n]=size(x);

tmp=zeros(m+masksize+2,n+masksize+2);
tmp(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1)=x_thr;
x_thr=tmp;

x1=zeros(m+masksize+2,n+masksize+2);
x1(1+masksize/2+1:m+masksize/2+1,1+masksize/2+1:n+masksize/2+1)=x;

Size_m=zeros(m+masksize+2,n+masksize+2);
Size_1=zeros(m+masksize+2,n+masksize+2);
Size_2=zeros(m+masksize+2,n+masksize+2);
GS_m=zeros(m+masksize+2,n+masksize+2);
GS_1=zeros(m+masksize+2,n+masksize+2);
GS_2=zeros(m+masksize+2,n+masksize+2);

G1_map=zeros(m+masksize+2,n+masksize+2);

%Even Mask


for i=1+masksize/2+1:m
for j=1+masksize/2+1:n
if x_thr(i,j)==1
        A=x1(i-masksize/2:i+masksize/2-1,j-masksize/2:j+masksize/2-1);
        if exist('ACF_store_PLICS','var')
            Size_loc_m=mean(nonzeros(ACF_store_PLICS{1}(i-masksize+1:i,j-masksize+1:j,1)));
            Size_loc_1=mean(nonzeros(ACF_store_PLICS{2}(i-masksize+1:i,j-masksize+1:j,1)));
            Size_loc_2=mean(nonzeros(ACF_store_PLICS{3}(i-masksize+1:i,j-masksize+1:j,1)));
        else
            Size_loc_m=0;
            Size_loc_1=0;
            Size_loc_2=0;
        end
        [RotProfile,OrientedProfiles,~]=ACFmean_test1_G1(A);
        %Insert calibration and size transformation here 
        [gf_m,Phase_m,~,Phase_S_m,~]=PhasorPlot_01_SingleProfile(RotProfile,MaxPhasorOrder);
        [gf_1,Phase_1,~,Phase_S_1,~]=PhasorPlot_01_SingleProfile(OrientedProfiles(1,:),MaxPhasorOrder);
        [gf_2,Phase_2,~,Phase_S_2,~]=PhasorPlot_01_SingleProfile(OrientedProfiles(2,:),MaxPhasorOrder);
        
        for k=1:MaxPhasorOrder
        %$£ The chosen value if the shifted phase value
        
                Phase_m(Phase_m<0)=0;
                Phase_1(Phase_1<0)=0;
                Phase_2(Phase_2<0)=0;
                
                Phase_S_m(Phase_S_m<0)=Phase_S_m(Phase_S_m<0)+pi;
                Phase_S_1(Phase_S_1<0)=Phase_S_1(Phase_S_1<0)+pi;
                Phase_S_2(Phase_S_2<0)=Phase_S_2(Phase_S_2<0)+pi;

                Phase_m=real((atanh((Phase_m-C0(1))./C0(2))./C0(3))+C0(4));
                Phase_1=real((atanh((Phase_1-C0(1))./C0(2))./C0(3))+C0(4));
                Phase_2=real((atanh((Phase_2-C0(1))./C0(2))./C0(3))+C0(4));
                Phase_S_m=real((atanh((Phase_S_m-C1(1))./C1(2))./C1(3))+C1(4))+Size_loc_m/PixelSize;
                Phase_S_1=real((atanh((Phase_S_1-C1(1))./C1(2))./C1(3))+C1(4))+Size_loc_1/PixelSize;
                Phase_S_2=real((atanh((Phase_S_2-C1(1))./C1(2))./C1(3))+C1(4))+Size_loc_2/PixelSize;

        %$£$£ Checks and corrects for saturated values

                Phase_m(real(Phase_m)>masksize/k)=masksize/k;
                Phase_S_m(real(Phase_S_m)>masksize/k)=masksize/k;
                Phase_1(real(Phase_1)>masksize/k)=masksize/k;
                Phase_S_1(real(Phase_S_1)>masksize/k)=masksize/k;
                Phase_2(real(Phase_2)>masksize/k)=masksize/k;
                Phase_S_2(real(Phase_S_2)>masksize/k)=masksize/k;
                Phase_S_m(isnan(Phase_S_m))=0;
                              
                
        end
        
  %$£ Store size and phasor values      

  if imag(round(Phase_S_m*2+1))~=0
      g=0;
  end
  
  if iseven(round(Phase_S_m*2+1))
     
      Lr=round(Phase_S_m*2+1)/2;
      Lcircle=2*Lr+1;
      Rcircle=Lr*Scale_reassign;
        Size_m(i-Lr:i+Lr,j-Lr:j+Lr)=Size_m(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*Phase_S_m.*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        Size_1(i-Lr:i+Lr,j-Lr:j+Lr)=Size_1(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*Phase_S_1.*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        Size_2(i-Lr:i+Lr,j-Lr:j+Lr)=Size_2(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*Phase_S_2.*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        GS_m(i-Lr:i+Lr,j-Lr:j+Lr)=GS_m(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*gf_m(1).*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        GS_1(i-Lr:i+Lr,j-Lr:j+Lr)=GS_1(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*gf_1(1).*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        GS_2(i-Lr:i+Lr,j-Lr:j+Lr)=GS_2(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*gf_2(1).*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        G1_map(i-Lr:i+Lr,j-Lr:j+Lr)=G1_map(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*RotProfile(1).*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        
        x_thr(i-Lr:i+Lr,j-Lr:j+Lr)=x_thr(i-Lr:i+Lr,j-Lr:j+Lr)-Circle(Lcircle,Rcircle);
        x_thr(x_thr<0)=0;
        else
      Lr=round(Phase_S_m*2)/2;    
      Lcircle=2*Lr+1;
      Rcircle=Lr*Scale_reassign;
        Size_m(i-Lr:i+Lr,j-Lr:j+Lr)=Size_m(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*Phase_S_m.*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        Size_1(i-Lr:i+Lr,j-Lr:j+Lr)=Size_1(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*Phase_S_1.*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        Size_2(i-Lr:i+Lr,j-Lr:j+Lr)=Size_2(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*Phase_S_2.*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        GS_m(i-Lr:i+Lr,j-Lr:j+Lr)=GS_m(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*gf_m(1).*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        GS_1(i-Lr:i+Lr,j-Lr:j+Lr)=GS_1(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*gf_1(1).*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        GS_2(i-Lr:i+Lr,j-Lr:j+Lr)=GS_2(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*gf_2(1).*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);
        G1_map(i-Lr:i+Lr,j-Lr:j+Lr)=G1_map(i-Lr:i+Lr,j-Lr:j+Lr)+Circle(Lcircle,Rcircle).*RotProfile(1).*x_thr(i-Lr:i+Lr,j-Lr:j+Lr);

        x_thr(i-Lr:i+Lr,j-Lr:j+Lr)=x_thr(i-Lr:i+Lr,j-Lr:j+Lr)-Circle(Lcircle,Rcircle);
        x_thr(x_thr<0)=0;
  end
  
end 
end  
end
    
GS_m=conj(GS_m);
GS_m(isnan(GS_m))=0;
GS_1=conj(GS_1);
GS_1(isnan(GS_1))=0;
GS_2=conj(GS_2);
GS_2(isnan(GS_2))=0;

ACF_store{1}=real(Size_m(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1))*PixelSize-3;
ACF_store{1}(:,:,2)=real(GS_m(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1))-3;
ACF_store{2}=real(Size_1(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1))*PixelSize-3;
ACF_store{2}(:,:,2)=real(GS_1(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1))-3;
ACF_store{3}=real(Size_2(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1))*PixelSize-3;
ACF_store{3}(:,:,2)=real(GS_2(2+masksize/2:m+masksize/2+1,2+masksize/2:n+masksize/2+1))-3;
ACF_store{4}=G1_map(masksize/2+2:end-masksize/2-1,masksize/2+2:end-masksize/2-1);
ACF_store{5}=(masksize*PixelSize)^2/pi./(ACF_store{1}(:,:,1).^2)./ACF_store{4};
ACF_store{5}(isinf(ACF_store{5}))=0;
ACF_store{5}(isnan(ACF_store{5}))=0;
end

%% PLICS Additional Functions
function C=Circle(DimMatrix)

mask_I = zeros(DimMatrix,DimMatrix);
xo = (DimMatrix+1)/2;
yo = (DimMatrix+1)/2;
r = (DimMatrix+1)/2;

% Make a mesh
dim_y = size(mask_I,1);
dim_x = size(mask_I,2);
[x,y] = meshgrid(-(xo-1):(dim_x-xo),-(yo-1):(dim_y-yo));

% Make the circular mask
C = (x.^2+y.^2)<=r^2;

end
function [B0,B1]=CalibrationParam_single_Circle01(mask)

load(strcat(pwd,'\PLICS - Functions\Calibration_circles.mat'))
A0=Calibration{2,(mask-6)/2};
A1=Calibration1{2,(mask-6)/2};

B0=real(A0(1,:));
B1=real(A1(1,:));
 
end
function [B0,B1]=CalibrationParam_single_new01(mask,PhasorOrder)

load(strcat(pwd,'\PLICS - Functions\Calibration.mat'))
A0=Calibration{2,(mask-6)/2};
A1=Calibration1{2,(mask-6)/2};

switch PhasorOrder
    case 1
        B0=real(A0(1,:));
        B1=real(A1(1,:));
    case 2
        B0=real(A0(2,:));
        B1=real(A1(2,:));
    case 3
        B0=real(A0(3,:));
        B1=real(A1(3,:));
end

end
function [Output,OrientedProfiles,Angles]=ACFmean_test0_G1(x)

[X,Y] = size(x);
Oversampling=0;
NumberOfAngles=60;
min_th_row=zeros(1,NumberOfAngles*2);

% calculate 2D ACF by FFT 
F=fft2(x);
ACF= F.*conj(F);
G=((sum(sum(x)))^2/X/Y);
ACF= ifft2(ACF);
ACF= fftshift(ACF)./G;
%ACF(ACF==absmax(ACF))=0;

if Oversampling>1
ACF=Oversample_Int(ACF,Oversampling);
end


[R, C]=size(ACF);

r0=R/2+1;
c0=C/2+1;

ACF1=flipud(ACF(2:r0,c0:end));
ACF2=ACF(r0:end,c0:end);
Radius=min(R/2,C/2);
ProfMat=zeros(NumberOfAngles*2,Radius);

for j=1:2
    if j==1
        y=ACF1';
    else
        y=ACF2;
    end
    
% CALCULATION OF ROTATIONAL MEAN
% Definition of angles
t=(pi/NumberOfAngles/2:pi/NumberOfAngles/2:pi/2);
   
% Matrix
y=y(1:Radius,1:Radius);
% Cycle between the 2nd and 2nd to last angles
[~, y1y]=size(y);

for i=1:NumberOfAngles
   rt=ceil(cos(t(i))*(1:Radius));
   ct=ceil(sin(t(i))*(1:Radius));
   profile=y((rt-1).*y1y+ct);

   if j==1
   ProfMat(NumberOfAngles+i,:)=profile;
   else
   ProfMat(i,:)=profile;
   end   
end

end

fwhm_1=ProfMat-absmin(ProfMat);
fwhm_1=abs(fwhm_1./absmax(fwhm_1)-0.5);
min_fw=min(fwhm_1,[],2);

for k=1:NumberOfAngles*2
    min_th_row1=ceil(find(fwhm_1(k,:)==min_fw(k)));
    if isempty(min_th_row1)
        min_th_row(k)=0;
    else
        min_th_row(k)=min_th_row1(1);
    end
end

min_th_row=find(min_th_row==min(min_th_row));
min_th=min_th_row(1)*pi/2/NumberOfAngles;

if min_th_row(1)<=NumberOfAngles
    max_th=min_th+pi/2;
    max_th_row=min_th_row(1)+NumberOfAngles;
else
    max_th=min_th-pi/2;
    max_th_row=min_th_row(1)-NumberOfAngles;
end

Prof2Mean=10;
ProfMat1=zeros(2*NumberOfAngles+Prof2Mean*2,Radius);
ProfMat1(1:Prof2Mean,:)=ProfMat(end-Prof2Mean+1:end,:);
ProfMat1(1+Prof2Mean:NumberOfAngles*2+Prof2Mean,:)=ProfMat;
ProfMat1(NumberOfAngles*2+Prof2Mean+1:NumberOfAngles*2+Prof2Mean*2,:)=ProfMat(1:Prof2Mean,:);
min_th_row=min_th_row+Prof2Mean;
max_th_row=max_th_row+Prof2Mean;

if Oversampling>1
min_fw_Profile=DSample1D(mean(ProfMat1(min_th_row(1)-Prof2Mean:min_th_row(1)+Prof2Mean,:),1),Oversampling);
max_fw_Profile=DSample1D(mean(ProfMat1(max_th_row(1)-Prof2Mean:max_th_row(1)+Prof2Mean,:),1),Oversampling);
Output=DSample1D(sum(ProfMat)/(2*NumberOfAngles),Oversampling);
else
min_fw_Profile=sum(ProfMat1(min_th_row(1)-Prof2Mean:min_th_row(1)+Prof2Mean,:),1)./(2*Prof2Mean+1);
max_fw_Profile=sum(ProfMat1(max_th_row(1)-Prof2Mean:max_th_row(1)+Prof2Mean,:),1)./(2*Prof2Mean+1);
Output=sum(ProfMat)./(2*NumberOfAngles);
end

Output=Output(2:end);
OrientedProfiles=min_fw_Profile(2:end);
OrientedProfiles(2,:)=max_fw_Profile(2:end);
Angles=[min_th,max_th];

end
function [gf,Phase,Mod,Phase_S,Mod_S]=PhasorPlot_01_SingleProfile(A,MaxPhasorOrder)
        
gf=fft(A);
gf=gf(MaxPhasorOrder+1)./gf(1);

Phase=-atan(imag(gf)./real(gf));
Mod=(sqrt(imag(gf).*imag(gf)+real(gf).*real(gf)));
Phase_S=-atan(imag(gf)./real(gf-0.5));
Mod_S=(sqrt(imag(gf).*imag(gf)+real(gf-0.5).*real(gf-0.5)));

end
 function cmap = map1
% Returns my custom colormap
cmap=colormap(jet);
cmap(1,:)=[00,00,000];
%cmap(65,:)=[0,0,0];
%cmap=flipud(cmap);
end
function b=median_nonzeros(a,dim)

if nargin<2
    dim=1;
end
[X,Y]=size(a);

if dim==1
    b=zeros(1,Y);
    for i=1:Y
        b(i)=median(nonzeros(a(:,i)));                
    end
end
if dim==2
    b=zeros(X,1);
    for i=1:X
        b(i)=median(nonzeros(a(i,:)));                
    end
end
end
function n_str=num2str_dig(n,dig)

n_str=num2str(n);
if numel(n_str)<dig

while numel(n_str)~=dig
    n_str=strcat('0',n_str); 
end

end
end
function b=std_nonzeros(a,~,dim)

if nargin<2
    dim=1;
end
[X,Y]=size(a);

if dim==1
    b=zeros(1,Y);
    for i=1:Y
        b(i)=std(nonzeros(a(:,i)));                
    end
end
if dim==2
    b=zeros(X,1);
    for i=1:X
        b(i)=std(nonzeros(a(i,:)));                
    end
end
end
function xlswrite_custom(filename,titles_cell,variables_cell,sheet,xlRange)

if iscell(titles_cell)
else
    tmp=titles_cell;
    titles_cell=cell(1);
    titles_cell{1}=tmp;
    clear tmp
end
if iscell(variables_cell)
else
    tmp=variables_cell;
    variables_cell=cell(1);
    variables_cell{1}=tmp;
    clear tmp
end
warning('off','MATLAB:xlswrite:AddSheet');
A=cell(1);
for i=1:length(titles_cell)
A{1,i}=titles_cell{i};

variable_tmp=variables_cell{i};
if ischar(variable_tmp)
A{2,i}=variable_tmp;
else
for j=1:length(variable_tmp)
    A{2+j,i}=variable_tmp(j);
end
end
end

if nargin<5
if nargin<4
sheet = 1;
end
xlRange = 'A1';
end
xlswrite(filename,A,sheet,xlRange)
end

   




