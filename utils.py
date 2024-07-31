import torch
import torch.fft as fft

# Fourier_filter関数は、入力テンソルxに対してフーリエ変換を行い、高周波成分をフィルタリングしてから逆フーリエ変換を行う
def Fourier_filter(x, threshold, scale):
    dtype = x.dtype
    x = x.type(torch.float32)
    # フーリエ変換を実行
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()

    # 高周波成分をフィルタリングするためのマスクを作成
    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # 逆フーリエ変換を実行
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    x_filtered = x_filtered.type(dtype)
    return x_filtered

# register_free_upblock2d関数は、モデルのアップサンプリングブロックをFreeU用に再定義する
def register_free_upblock2d(model, b1=1.2, b2=1.4, s1=0.9, s2=0.2):
    # アップサンプリングブロックの順方向計算を定義する内部関数
    def up_forward(self):
        def forward(hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
            for resnet in self.resnets:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                
                # FreeUの高周波フィルタリングとスケーリングを適用
                if hidden_states.shape[1] == 1280:
                    hidden_states[:,:640] = hidden_states[:,:640] * self.b1
                    res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=self.s1)
                if hidden_states.shape[1] == 640:
                    hidden_states[:,:320] = hidden_states[:,:320] * self.b2
                    res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=self.s2)

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states
        
        return forward
    
    # モデルの全てのアップサンプリングブロックに対して、FreeUの順方向計算を設定
    for upsample_block in model.unet.up_blocks:
        if upsample_block.__class__.__name__ == "UpBlock2D":
            upsample_block.forward = up_forward(upsample_block)
            setattr(upsample_block, 'b1', b1)
            setattr(upsample_block, 'b2', b2)
            setattr(upsample_block, 's1', s1)
            setattr(upsample_block, 's2', s2)

# register_free_crossattn_upblock2d関数は、モデルのクロスアテンションアップサンプリングブロックをFreeU用に再定義する
def register_free_crossattn_upblock2d(model, b1=1.2, b2=1.4, s1=0.9, s2=0.2):
    # クロスアテンションアップサンプリングブロックの順方向計算を定義する内部関数
    def up_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple,
            temb = None,
            encoder_hidden_states = None,
            cross_attention_kwargs = None,
            upsample_size = None,
            attention_mask = None,
            encoder_attention_mask = None,
        ):
            for resnet, attn in zip(self.resnets, self.attentions):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # FreeUの高周波フィルタリングとスケーリングを適用
                if hidden_states.shape[1] == 1280:
                    hidden_states[:,:640] = hidden_states[:,:640] * self.b1
                    res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=self.s1)
                if hidden_states.shape[1] == 640:
                    hidden_states[:,:320] = hidden_states[:,:320] * self.b2
                    res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=self.s2)

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )[0]

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states
        
        return forward
    
    # モデルの全てのクロスアテンションアップサンプリングブロックに対して、FreeUの順方向計算を設定
    for upsample_block in model.unet.up_blocks:
        if upsample_block.__class__.__name__ == "CrossAttnUpBlock2D":
            upsample_block.forward = up_forward(upsample_block)
            setattr(upsample_block, 'b1', b1)
            setattr(upsample_block, 'b2', b2)
            setattr(upsample_block, 's1', s1)
            setattr(upsample_block, 's2', s2)