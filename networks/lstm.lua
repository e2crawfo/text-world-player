require 'nn'
require 'rnn'  -- IMP: dont use LSTM package from nnx - buggy
require 'nngraph'

--require 'cunn'
-- IMP if args is not passed, global 'args' are taken.


-- overriding LSTM factory functions
LSTM = nn.LSTM

-- incoming is {input(t), output(t-1), cell(t-1)}
function LSTM:buildModel()
   -- build components
   self.cellLayer = self:buildCell()
   self.outputGate = self:buildOutputGate()
   -- assemble
   local concat = nn.ConcatTable()
   local concat2 = nn.ConcatTable()
   local input_transform = nn.ParallelTable()
   input_transform:add(EMBEDDING)
   input_transform:add(nn.Identity())
   input_transform:add(nn.Identity())

   concat2:add(nn.SelectTable(1)):add(nn.SelectTable(2))
   concat:add(concat2):add(self.cellLayer)
   local model = nn.Sequential()
   model:add(input_transform)
   model:add(concat)
   -- output of concat is {{input, output}, cell(t)},
   -- so flatten to {input, output, cell(t)}
   model:add(nn.FlattenTable())
   local cellAct = nn.Sequential()
   cellAct:add(nn.SelectTable(3))
   cellAct:add(nn.Tanh())
   local concat3 = nn.ConcatTable()
   concat3:add(self.outputGate):add(cellAct)
   local output = nn.Sequential()
   output:add(concat3)
   output:add(nn.CMulTable())
   -- we want the model to output : {output(t), cell(t)}
   local concat4 = nn.ConcatTable()
   concat4:add(output):add(nn.SelectTable(3))
   model:add(concat4)
   print(model)
   return model
end


function LSTM:updateOutput(input)
   local prevOutput, prevCell
   if self.step == 1 then
      prevOutput = self.zeroTensor
      prevCell = self.zeroTensor

      -- since we have batches of symbols, we need to do this explicitly
      self.zeroTensor:resize(input:size(1), self.outputSize):zero()

      self.outputs[0] = self.zeroTensor
      self.cells[0] = self.zeroTensor
   else
      -- previous output and cell of this module
      prevOutput = self.output
      prevCell = self.cell
   end

   -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
   local output, cell
   if self.train ~= false then
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      -- the actual forward propagation
      output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
   else
      output, cell = unpack(self.recurrentModule:updateOutput{input, prevOutput, prevCell})
   end

   if self.train ~= false then
      local input_ = self.inputs[self.step]
      self.inputs[self.step] = self.copyInputs
         and rnn.recursiveCopy(input_, input)
         or rnn.recursiveSet(input_, input)
   end

   self.outputs[self.step] = output
   self.cells[self.step] = cell

   self.output = output
   self.cell = cell

   self.step = self.step + 1
   self.gradParametersAccumulated = false
   -- note that we don't return the cell, just the output
   return self.output
end


local function make_lstm(hist_len, gpu)
    n_hid = EMBEDDING.weight:size(2)
    local l  = LSTM(n_hid, n_hid)

    local lstm_seq = nn.Sequential()
    lstm_seq:add(nn.Sequencer(l))

    -- Mean pooling
    lstm_seq:add(nn.CAddTable())
    lstm_seq:add(nn.Linear(n_hid, n_hid))
    lstm_seq:add(nn.Rectifier())

    LSTM_MODEL = lstm_seq

    local parallel_flows = nn.ParallelTable()
    for f=1, hist_len do
        if f > 1 then
            parallel_flows:add(lstm_seq:clone("weight","bias","gradWeight", "gradBias"))
        else
            parallel_flows:add(lstm_seq) -- TODO share 'weight' and 'bias'
        end
    end

    local lstm = nn.Sequential()
    lstm:add(parallel_flows)
    lstm:add(nn.JoinTable(2))

    if gpu == 1 then
        lstm:cuda()
    end

    return lstm, n_hid * args.hist_len
end

lstm = {
    make_lstm=make_lstm
}

return lstm
