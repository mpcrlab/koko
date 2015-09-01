%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
%
% Machine Perception and Cognitive Robotics Laboratory
%
%     Center for Complex Systems and Brain Sciences
%
%              Florida Atlantic University
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
% Locally Competitive Algorithms Demonstration
% Using natural images data, see:
% Rozell, Christopher J., et al.
% "Sparse coding via thresholding and
% local competition in neural circuits."
% Neural computation 20.10 (2008): 2526-2563.
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function HahnLCA_Dictionary_RGB_CIFAR_Block


clear all
close all
clc
beep off
% rng('default')
% rng(1)


%cd /Users/williamedwardhahn/Documents/MATLAB/cifar


labelnames={'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'};


XX=[];

for k = 1:10
    
    
    cd patches
    load(strjoin(['HahnColorPatchesCIFAR_5k_',labelnames(k),'.mat'],''))
    cd ..
    
    X=patches;
    
    X = bsxfun(@minus,X,mean(X)); %remove mean
    fX = fft(fft(X,[],2),[],3); %fourier transform of the images
    spectr = sqrt(mean(abs(fX).^2)); %Mean spectrum
    X = ifft(ifft(bsxfun(@times,fX,1./spectr),[],2),[],3); %whitened X
    
    XX=[XX sqrt(0.1)*X/sqrt(mean(var(X)))];
    
    size(XX)
    
    
    
end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    s=0.01;
    patch_size=32^2*3;
    neurons=3600;
    batch_size=1024;
    
    W = randn(patch_size, neurons);
    
    for j=1:800
        j
        
        r=randperm(size(XX,2));
        
        X=XX(:,r(1:batch_size));
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        W = W*diag(1./sqrt(sum(W.^2,1)));
        
        b = W'*X;
        G = W'*W - eye(neurons);
        
        u = zeros(neurons,batch_size);
        
        for i =1:64
            
            if j < 200
                a=u.*(abs(u) > s);
            else
                switch floor(j/300)+1%mod(j,7)+1%randi(8)
                    case 1
                        a=blocksparse_vec(u,30);
                    case 2
                        a=blocksparse_vec(u,20);
                    case 3
                        a=blocksparse_vec(u,15);
                    case 4
                        a=blocksparse_vec(u,12);
                    case 5
                        a=blocksparse_vec(u,10);
                    case 6
                        a=blocksparse_vec(u,6);
                    case 7
                        a=blocksparse_vec(u,5);
                    case 8
                        a=blocksparse_vec(u,4);
                    case 9
                        a=blocksparse_vec(u,3);
                    case 10
                        a=blocksparse_vec(u,2);
%                     case 8
%                         a=u.*(abs(u) > s);
                        
                end
            end
            
            u = 0.9 * u + 0.01 * (b - G*a);
            
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        W = W + (1/batch_size)*((X-W*a)*a');
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        
        %imagesc(filterplotcolor(W')), drawnow()
        
        
        
    end
    
    save('MPCR_CIFAR_Dictionary_Multi.mat','W')
    
end

















function XX = blocksparse_vec(XX,blocksize)

s=size(XX(:,1));
n=sqrt(s(1));

blocks=n/blocksize;

A = reshape(1:n^2,[n n])';
B = im2col1(A,[n/blocks n/blocks]);

for i=1:size(XX,2)
    
    X=reshape(XX(:,i),[n n]);
    
    [m1 m2]=max(sum(abs(X(B)),1));
    
    D=0*B;
    
    D(:,m2)=X(B(:,m2));
    
    XX(:,i)=reshape(im2col1(D,[blocksize blocks]),s);
    
end

end







function E = im2col1(A,blocksize)

r = blocksize(1);
c = blocksize(2);
e = r*c;

B = zeros(size(A,1)+((mod(size(A,1),r)~=0)*(r - mod(size(A,1),r))),size(A,2)+((mod(size(A,2),c)~=0)*(c - mod(size(A,2),c))));
B(1:size(A,1),1:size(A,2)) = A;

C = reshape(B,r,size(B,1)/r,[]);
D = reshape(permute(C,[1 3 2]),size(C,1)*size(C,3),[]);
E = reshape(permute(reshape(D,e,size(D,1)/e,[]),[1 3 2]),e,[]);

return;

end














function [D] = filterplotcolor(W)

Dr=filterplot(W(:,1:size(W,2)/3));
Dg=filterplot(W(:,size(W,2)/3+1:2*size(W,2)/3));
Db=filterplot(W(:,2*size(W,2)/3+1:end));
D=zeros(size(Dr,1),size(Dr,2),3);
D(:,:,1)=Dr;
D(:,:,2)=Db;
D(:,:,3)=Dg;
D = D - min(D(:));
D = D / max(D(:));

end









function [D] = filterplot(X)

[m,n] = size(X);
w = round(sqrt(n));
h = (n / w);
c = floor(sqrt(m));
r = ceil(m / c);
p = 1;
D = - ones(p + r * (h + p),p + c * (w + p));
k = 1;
for j = 1:r
    for i = 1:c
        D(p + (j - 1) * (h + p) + (1:h), p + (i - 1) * (w + p) + (1:w)) = reshape(X(k, :), [h, w]) / max(abs(X(k, :)));
        k = k + 1;
    end
end

end



