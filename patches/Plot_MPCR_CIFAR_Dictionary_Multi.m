function Plot_MPCR_CIFAR_Dictionary_Multi


load('MPCR_CIFAR_Dictionary_Multi.mat')
imagesc(filterplotcolor(W')), drawnow()

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



