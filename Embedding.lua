require 'nn'
require 'utils'

do
    local Embedding, _ = torch.class('Embedding', 'nn.LookupTable')

    -- override (zero out NULL INDEX)
    function Embedding:updateOutput(input)
        self:backCompatibility()
        input = self:makeInputContiguous(input)
        if input:dim() == 1 then
           self.output:index(self.weight, 1, input)
        elseif input:dim() == 2 then
           self.output:index(self.weight, 1, input:view(-1))
           self.output = self.output:view(input:size(1), input:size(2), self.weight:size(2))
        else
           error("input must be a vector or matrix")
        end

        --zero out NULL_INDEX
        local output = self.output:clone()
        for i=1, input:size(1) do
            if input[i] == #symbols+1 then
                output[i]:mul(0)
            end
        end

        self.output = output

        return self.output
    end

    function Embedding:setWordVecs(symbols, vector_function)
        self.weight[#symbols+1]:mul(0) --zero out NULL INDEX vector

        for i=1, #symbols do
            self.weight[i] = vector_function(self.weight:size(2), symbols[i])
            print("wordvec", symbols[i], self.weight[i])
        end
    end
end
