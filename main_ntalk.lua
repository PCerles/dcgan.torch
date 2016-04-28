require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'cunn'
package.path = package.path .. ";/home/vashishtm/ImageGen/neuraltalk2/?.lua;/home/vashishtm/ImageGen/neuraltalk2/model/?.t7"
util = paths.dofile('util.lua')
--ntalk_util = require('external')
--local ntalk_model =  '../neuraltalk2/model/karpathymodel.t7_cpu.t7'
--local ntalk_protos = ntalk_util.getProtos(ntalk_model, -1)
nngraph.setDebug(true)
opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nz = 1024,               -- #  of dim for Z
   ncond = 1024, 	         -- #  of dim for C
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 100,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 2,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
   noise = 'normal',       -- uniform / normal
   conditional = true,
   checkpoint = 0
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz
local ncond = opt.ncond
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0
local conditional = opt.conditional

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
local JoinTable = nn.JoinTable

local nets = {}

function nets.generativeNet(ngf, noise_size, cond_size, conditional, out_size)
    -- local noise = nn.Identity()()
    -- local cond = nn.Identity()()
    
    -- input is Z + C, going into a convolution
    -- concatenation table
    -- local concat1 = JoinTable(2)({noise, cond})
    
    
    --concatenate before calling net.forward
    local data = nn.Identity()()

    local conv1 = SpatialFullConvolution(nz+ncond, ngf * 8, 4, 4)(data)
    local bn1 = SpatialBatchNormalization(ngf * 8)(conv1)
    local relu1 = nn.ReLU(true)(bn1)

    --concat2 = JoinTable(1)({relu1,condInput})
    --conv2 = (SpatialFullConvolution(ngf * 8 + ncond, ngf * 4, 4, 4, 2, 2, 1, 1))(concat2)
    local conv2 = (SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))(relu1)
    local bn2 = (SpatialBatchNormalization(ngf * 4))(conv2)
    local relu2 = nn.ReLU(true)(bn2)

    --concat3 = JoinTable(1)({relu2,condInput})
    --conv3 = SpatialFullConvolution(ngf * 4 + ncond, ngf * 2, 4, 4, 2, 2, 1, 1)(concat3)
    local conv3 = (SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))(relu2)
    local bn3 = SpatialBatchNormalization(ngf * 2)(conv3)
    local relu3 = nn.ReLU(true)(bn3)

    --concat4 = JoinTable(1)({relu3,condInput})
    --conv4 = SpatialFullConvolution(ngf * 2 + ncond, ngf, 4, 4, 2, 2, 1, 1)(concat4)
    local conv4 = SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1)(relu3)
    local bn4 = SpatialBatchNormalization(ngf)(conv4)
    local relu4 = nn.ReLU(true)(bn4)

    --concat5 = JoinTable(1)({relu4,condInput})
    --conv5 = SpatialFullConvolution(ngf + ncond, nc, 4, 4, 2, 2, 1, 1)(concat5)
    local conv5 = SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1)(relu4)
    local output = nn.Tanh()(conv5)

    local netG = nn.gModule({data}, {output})
	  return netG
end

function nets.discriminativeNet(ndf, in_size, conditional)tD
    local netD = nn.Sequential()
    -- Discriminative Network
    -- input is (nc) x 64 x 64
    netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 32 x 32
    netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 16 x 16
    netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 8 x 8
    netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 4 x 4
    netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
    netD:add(nn.Sigmoid())
    -- state size: 1 x 1 x 1
    netD:add(nn.View(1):setNumInputDims(3))
    -- state size: 1
    return netD
end

if opt.checkpoint == 0 then
	netG = nets.generativeNet(ngf, noise_size, cond_size, conditional, out_size)
	netD = nets.discriminativeNet(ndf, in_size, ncond, conditional)
	netG:apply(weights_init)
	netD:apply(weights_init)
else
	netG = util.load('checkpoints/' .. opt.name .. '_' .. opt.checkpoint .. '_net_G.t7', opt.gpu)
	netD = util.load('checkpoints/' .. opt.name .. '_' .. opt.checkpoint .. '_net_D.t7', opt.gpu)
end
local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local cond = torch.Tensor(opt.batchSize, ncond, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda(); cond = cond:cuda();
   netG = util.cudnn(netG);     netD = util.cudnn(netD)
   netD:cuda();           netG:cuda();           criterion:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise:clone()

if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

local fDxCond = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real, captions,caption_vecs = data:getBatch()
   data_tm:stop()

   input:copy(real)
   label:fill(real_label)

   local output = netD:forward(input) -- run with image + conditional info
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end

   -- make a generated image
   local joinData = torch.cat(noise, cond, 2)
   local fake = netG:forward(joinData)

   input:copy(fake)

   label:fill(fake_label)

   -- run discriminative net on generated image
   local output = netD:forward(input)
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   local ntalk_loss = ntalk_util.getLoss(ntalk_protos, fake, captions)
   print(ntalk_loss)

   errD = errD_real + errD_fake

   return errD, gradParametersD
end

local fGxCond = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput({input,cond}, df_do)
   local df_di = df_dg[1]
   local joinData = torch.cat(noise, cond, 2)
   netG:backward(joinData, df_di)
   return errG, gradParametersG
end
-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDxCond, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGxCond, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          --local data = torch.cat(noise_vis, cond, 2)
          --local fake = netG:forward(data)
          --local real = data:getBatch()
          --disp.image(fake, {win=opt.display_id, title=opt.name})
          --disp.image(real, {win=opt.display_id * 3, title=opt.name})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   print(netG)
   if epoch % 25 == 0 then
	real_epoch = opt.checkpoint + epoch
   	util.save('checkpoints/' .. opt.name .. '_' .. real_epoch .. '_net_G.t7', netG, opt.gpu)
   	util.save('checkpoints/' .. opt.name .. '_' .. real_epoch .. '_net_D.t7', netD, opt.gpu)
   end
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
