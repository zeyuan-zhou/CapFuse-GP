import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import os
import copy
import gc
import argparse


class Config:

    TARGET_TRAIT = 'YLD'
    USE_PCA_CORRECTION = True
    N_PCA_COMPONENTS = 3

    TEST_SIZE = 0.10
    N_REPEATS = 5
    N_FOLDS = 5
    SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="CapFuse-GP ensemble genomic prediction"
    )


    parser.add_argument("--data", dest="CSV_PATH", required=True,
                        help="Path to the input CSV file")
    parser.add_argument("--trait", dest="TARGET_TRAIT", default=Config.TARGET_TRAIT,
                        help="Target phenotype column")
    parser.add_argument("--gpu-id", dest="GPU_ID", type=int, default=0,
                        help="CUDA GPU id")


    parser.add_argument("--k-best-snps", dest="K_BEST_SNPS", type=int, required=True)
    parser.add_argument("--batch-size", dest="BATCH_SIZE", type=int, required=True)
    parser.add_argument("--epochs", dest="EPOCHS", type=int, required=True)
    parser.add_argument("--patience", dest="PATIENCE", type=int, required=True)

    parser.add_argument("--lr", dest="LEARNING_RATE", type=float, required=True)
    parser.add_argument("--weight-decay", dest="WEIGHT_DECAY", type=float, required=True)

    parser.add_argument("--bb1-dropout", dest="BB1_DROPOUT", type=float, required=True)
    parser.add_argument("--bb2-dropout", dest="BB2_DROPOUT", type=float, required=True)
    parser.add_argument("--fusion-dropout", dest="FUSION_DROPOUT", type=float, required=True)
    parser.add_argument("--fusion-hidden-dim", dest="FUSION_HIDDEN_DIM",
                        type=int, required=True)


    parser.add_argument("--bb1-blocks", dest="BB1_NUM_BLOCKS",
                        type=int, nargs=4, required=True,
                        metavar=("B1", "B2", "B3", "B4"))
    parser.add_argument("--bb2-blocks", dest="BB2_NUM_BLOCKS",
                        type=int, nargs=4, required=True,
                        metavar=("B1", "B2", "B3", "B4"))

    cli_args = parser.parse_args()


    cfg = Config()
    for key, value in vars(cli_args).items():
        setattr(cfg, key, value)

    return cfg


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetSE_1D(nn.Module):
    def __init__(self, num_blocks, dropout_rate, input_channels=3, num_classes=1):
        super(ResNetSE_1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=15, stride=4, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResidualSEBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(ResidualSEBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(ResidualSEBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(ResidualSEBlock, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(512, num_classes))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _get_features(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x);
        x = self.layer2(x);
        x = self.layer3(x);
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        features = self._get_features(x)
        return self.fc(features)


class FusionResNet(nn.Module):
    def __init__(self, backbone1, backbone2, hidden_dim, dropout, feature_dim=512):
        super(FusionResNet, self).__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        feat1 = self.backbone1._get_features(x)
        feat2 = self.backbone2._get_features(x)
        fused = torch.cat((feat1, feat2), dim=1)
        return self.fusion_head(fused)


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    pcc = np.corrcoef(y_true, y_pred)[0, 1] if len(np.unique(y_pred)) > 1 else 0.0
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "PCC": pcc}


class G2PDataset(Dataset):
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor
        self.y = y_tensor

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def encode_genotypes(df_segment):
    gmp = df_segment.replace({-1: 0, 0: 1, 1: 2}).astype(np.int8)
    ns, snp = gmp.shape
    gts = torch.zeros((ns, snp, 3), dtype=torch.float32)
    gts[np.arange(ns)[:, None], np.arange(snp), gmp.values] = 1
    return gts.permute(0, 2, 1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds = 0, []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item() * X.size(0)
            preds.append(output.cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(preds)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Trait: {args.TARGET_TRAIT}")
    print(f"Strategy: Independent Test Set + Ensemble Prediction (Avg of {args.N_FOLDS} Inner Models)")

    if not os.path.exists(args.CSV_PATH):
        print("Data file not found.");
        exit()


    full_df = pd.read_csv(args.CSV_PATH)
    meta_cols = ['Entry', 'GrW', 'GrL', '1000GW', 'PH', 'FlgLL', 'FlgLW', 'FlgLA', 'PnN', 'PnL', 'YLD', 'YPP']
    snp_cols = [c for c in full_df.columns if c not in meta_cols]

    X_raw = full_df[snp_cols].apply(lambda c: np.select([c > 0.5, c < -0.5], [1., -1.], default=0.), axis=0)
    y_raw = full_df[args.TARGET_TRAIT].values.reshape(-1, 1)

    valid_mask = ~np.isnan(y_raw.flatten())
    X_raw, y_raw = X_raw[valid_mask], y_raw[valid_mask]
    indices = np.arange(len(X_raw))

    final_test_results = []


    for repeat in range(args.N_REPEATS):
        outer_seed = args.SEED + repeat
        print(f"\n" + "=" * 60)
        print(f"INDEPENDENT EXPERIMENT {repeat + 1}/{args.N_REPEATS} (Seed {outer_seed})")


        train_val_idx, test_idx = train_test_split(indices, test_size=args.TEST_SIZE, random_state=outer_seed)

        X_pool, y_pool = X_raw.iloc[train_val_idx], y_raw[train_val_idx]
        X_test_holdout, y_test_holdout = X_raw.iloc[test_idx], y_raw[test_idx]


        ensemble_predictions = []


        kf = KFold(n_splits=args.N_FOLDS, shuffle=True, random_state=outer_seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_pool)):
            print(f"\n  [Exp {repeat + 1}] Training Inner Model {fold + 1}/{args.N_FOLDS} ...")


            X_train, X_val = X_pool.iloc[train_idx], X_pool.iloc[val_idx]
            y_train, y_val = y_pool[train_idx], y_pool[val_idx]


            y_train_for_sel = y_train.ravel()
            if args.USE_PCA_CORRECTION:
                pca_subset = X_train.sample(n=min(5000, X_train.shape[1]), axis=1, random_state=outer_seed)
                pca = PCA(n_components=args.N_PCA_COMPONENTS)
                PCs = pca.fit_transform(pca_subset)
                reg = LinearRegression().fit(PCs, y_train)
                y_train_for_sel = (y_train - reg.predict(PCs)).ravel()

            selector = SelectKBest(f_regression, k=min(args.K_BEST_SNPS, X_train.shape[1]))
            selector.fit(X_train, y_train_for_sel)
            support = selector.get_support()

            X_train_s = X_train.iloc[:, support]
            X_val_s = X_val.iloc[:, support]
            X_test_holdout_s = X_test_holdout.iloc[:, support]


            scaler_y = MinMaxScaler()
            y_train_s = scaler_y.fit_transform(y_train)
            y_val_s = scaler_y.transform(y_val)
            y_test_holdout_s = scaler_y.transform(y_test_holdout)


            train_loader = DataLoader(
                G2PDataset(encode_genotypes(X_train_s), torch.tensor(y_train_s, dtype=torch.float32)),
                batch_size=args.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(G2PDataset(encode_genotypes(X_val_s), torch.tensor(y_val_s, dtype=torch.float32)),
                                    batch_size=args.BATCH_SIZE)
            test_loader = DataLoader(
                G2PDataset(encode_genotypes(X_test_holdout_s), torch.tensor(y_test_holdout_s, dtype=torch.float32)),
                batch_size=args.BATCH_SIZE)


            backbone1 = ResNetSE_1D(num_blocks=args.BB1_NUM_BLOCKS, dropout_rate=args.BB1_DROPOUT).to(device)
            backbone2 = ResNetSE_1D(num_blocks=args.BB2_NUM_BLOCKS, dropout_rate=args.BB2_DROPOUT).to(device)
            model = FusionResNet(backbone1, backbone2, feature_dim=512, hidden_dim=args.FUSION_HIDDEN_DIM,
                                 dropout=args.FUSION_DROPOUT).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)
            criterion = nn.MSELoss()

            best_model_wts = copy.deepcopy(model.state_dict())
            min_val_loss = float('inf')
            no_improve = 0

            for epoch in range(args.EPOCHS):
                train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, _ = evaluate(model, val_loader, criterion, device)

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= args.PATIENCE: break


            model.load_state_dict(best_model_wts)
            _, preds_scaled = evaluate(model, test_loader, criterion, device)


            preds_real_fold = scaler_y.inverse_transform(preds_scaled)
            ensemble_predictions.append(preds_real_fold)

            print(f"    -> Model {fold + 1} trained (Val MSE: {min_val_loss:.4f}). Added to Ensemble.")

            del model, optimizer, backbone1, backbone2
            torch.cuda.empty_cache();
            gc.collect()


        all_preds = np.array(ensemble_predictions)
        avg_pred = np.mean(all_preds, axis=0)


        ensemble_metrics = calculate_metrics(y_test_holdout, avg_pred)
        print(
            f"  [Exp {repeat + 1} Result] Ensemble PCC: {ensemble_metrics['PCC']:.4f} | MSE: {ensemble_metrics['MSE']:.4f}")
        final_test_results.append(ensemble_metrics)


    print("\n" + "=" * 60)
    print("FINAL ENSEMBLE RESULTS")
    print("=" * 60)
    df_res = pd.DataFrame(final_test_results)
    print(df_res)
    print("-" * 30)
    print(f"Mean PCC: {df_res['PCC'].mean():.4f}  (Std: {df_res['PCC'].std():.4f})")
    print(f"Mean MSE: {df_res['MSE'].mean():.4f}  (Std: {df_res['MSE'].std():.4f})")

    df_res.to_csv("DSE-FusionNet_20repeats.csv", index=False)
