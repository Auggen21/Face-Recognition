% clc
clear all
close all

load('trainedClassifier.mat')

I1=imread('att_faces\s2\3.pgm');
ginp=I1;
[m,n] = size(ginp);

blocksize=16;
countj = 0;
for i = 1:blocksize:m-(blocksize-1)
   %counti = counti + 1;
   %countj = 0;
   for j = 1:blocksize:n-(blocksize-1)
        countj = countj + 1;
        Blocks{countj} = ginp(i:i+(blocksize-1),j:j+(blocksize-1));
   end
end

%HOG
cm=1;
for p=1:countj
    hogfeat{p} = extractHOGFeatures(Blocks{p});
end

halfsize=blocksize/2;
%LDB
[sR1,sC1,eR1,eC1] = deal(1,1,halfsize,halfsize);%startingRow, startingColumn, endingRow, endingColumn
[sR2,sC2,eR2,eC2] = deal(1,halfsize+1,halfsize,blocksize);
[sR3,sC3,eR3,eC3] = deal(halfsize+1,1,blocksize,halfsize);
[sR4,sC4,eR4,eC4] = deal(halfsize+1,halfsize+1,blocksize,blocksize);

mX=blocksize/4;
mY=blocksize/4;

%
% feat1="";

for p=1:countj

    J = integralImage(Blocks{p});
    regionSum1 = J(eR1+1,eC1+1) - J(eR1+1,sC1) - J(sR1,eC1+1) + J(sR1,sC1);%i4-i3-i2+i1
    regionSum2 = J(eR2+1,eC2+1) - J(eR2+1,sC2) - J(sR2,eC2+1) + J(sR2,sC2);
    regionSum3 = J(eR3+1,eC3+1) - J(eR3+1,sC3) - J(sR3,eC3+1) + J(sR3,sC3);
    regionSum4 = J(eR4+1,eC4+1) - J(eR4+1,sC4) - J(sR4,eC4+1) + J(sR4,sC4);
    avg1=regionSum1/(blocksize*blocksize);
    avg2=regionSum2/(blocksize*blocksize);
    avg3=regionSum3/(blocksize*blocksize);
    avg4=regionSum4/(blocksize*blocksize);

    dx1=regionSum1-2*(J(eR1+1,mX+1) - J(eR1+1,sC1) - J(sR1,eC1+1) + J(sR1,sC1));%i7-i3-i2+i1
    dy1=regionSum1-2*(J(mY+1,eC1+1) - J(mX+1,sC1) - J(sR1,eC1+1) + J(sR1,sC1));%i6-i8-i2+i1

    dx2=regionSum2-2*(J(eR2+1,mX+1) - J(eR2+1,sC2) - J(sR2,eC2+1) + J(sR2,sC2));
    dy2=regionSum2-2*(J(mY+1,eC2+1) - J(mX+1,sC2) - J(sR2,eC2+1) + J(sR2,sC2));

    dx3=regionSum3-2*(J(eR3+1,mX+1) - J(eR3+1,sC3) - J(sR3,eC3+1) + J(sR3,sC3));
    dy3=regionSum3-2*(J(mY+1,eC3+1) - J(mX+1,sC3) - J(sR3,eC3+1) + J(sR3,sC3));

    dx4=regionSum4-2*(J(eR4+1,mX+1) - J(eR4+1,sC4) - J(sR4,eC4+1) + J(sR4,sC4));
    dy4=regionSum4-2*(J(mY+1,eC4+1) - J(mX+1,sC4) - J(sR4,eC4+1) + J(sR4,sC4));


    avgf=[avg1 avg2 avg3 avg4];
    dx=[dx1 dx2 dx3 dx4];
    dy=[dy1 dy2 dy3 dy4];
    %comparisons

    k=1;
     feat2="";
    for mmm=1:4
        for nnn=mmm:4
            feat1="";
            if (avgf(mmm)>avgf(nnn))&(mmm~=nnn)
                feat1=feat1+"1";
            else
                feat1=feat1+"0";
            end

            if (dx(mmm)>dx(nnn))&(mmm~=nnn)
                feat1=feat1+"1";
            else
                feat1=feat1+"0";
            end

            if (dy(mmm)>dy(nnn))&(mmm~=nnn)
                feat1=feat1+"1";
            else
                feat1=feat1+"0";
            end

            if mmm~=nnn
                feat2=feat2+feat1;
            end

        end

    end

    feat2=char(feat2);
    ldbfeatmat=zeros(1,size(feat2,2));
    for mn=1:size(feat2,2)
        ldbfeatmat(mn)=str2num(feat2(mn));
    end

    ldbfeat{p}=ldbfeatmat;

end

predMat=[];
for p=1:countj
predMat=[predMat hogfeat{p} ldbfeat{p}];
end

 yfit = trainedClassifier.predictFcn(predMat);

disp('The person is s',yfit);