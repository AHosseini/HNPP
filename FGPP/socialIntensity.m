function I = socialIntensity(incomingEdges,eventMatrix,t,w,g,user,product)
    I = 0;
    [~,V,TAU] = find(incomingEdges{user});
    %{
    if (user == 5)
        disp(V);
        disp(TAU);
    end
    %}
    for i=1:length(V)
        v = V(i);
        tau = TAU(i);
        for j=1:length(eventMatrix{v,product})
            ti = eventMatrix{v,product}(j);
            if (ti >= t)
                break;
            end
            I = I+tau*g(t-ti,w);
        end
    end
end