import React, { useState, useEffect } from 'react';
import {
  listAlertRules,
  saveAlertRule,
  dispatchTestAlert,
  type AlertRule,
} from '../../services/enterpriseService';
import { useNotification } from '../../context/NotificationContext';

type Props = {
  isOpen: boolean;
  onClose: () => void;
};

export default function AlertRuleConfigModal({ isOpen, onClose }: Props) {
  const { notify } = useNotification();

  const [rules, setRules] = useState<AlertRule[]>([]);
  const [name, setName] = useState('Data Drift & Model Health Channel');
  const [channelType, setChannelType] = useState('slack');
  const [webhookUrl, setWebhookUrl] = useState('https://example.com/webhooks/slack-incoming');
  const [emailRecipient, setEmailRecipient] = useState('alerts@company.com');
  const [triggerDrift, setTriggerDrift] = useState(true);
  const [triggerDegradation, setTriggerDegradation] = useState(true);
  const [loading, setLoading] = useState(false);

  const loadRules = async () => {
    try {
      const data = await listAlertRules();
      setRules(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    if (isOpen) loadRules();
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSaveRule = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await saveAlertRule({
        name,
        channel_type: channelType,
        webhook_url: webhookUrl,
        email_recipient: emailRecipient,
        trigger_on_drift: triggerDrift,
        trigger_on_degradation: triggerDegradation,
        enabled: true,
      });
      notify('Alert notification channel saved successfully!', 'success');
      loadRules();
    } catch (e: any) {
      notify(e.message || 'Failed to save alert channel', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleSendTest = async (ruleId?: string) => {
    setLoading(true);
    try {
      const res = await dispatchTestAlert({
        rule_id: ruleId,
        title: 'Observability Test Dispatch',
        message: 'Auto Data Analyst continuous monitoring test payload delivered successfully.',
        severity: 'INFO',
      });
      notify(`Test notification dispatched (${res[0]?.delivery_status || 'DELIVERED'})!`, 'success');
    } catch (e: any) {
      notify(e.message || 'Failed to send test alert', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.65)',
        backdropFilter: 'blur(4px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.5rem',
      }}
    >
      <div
        className="glass-card"
        style={{
          width: '100%',
          maxWidth: '650px',
          backgroundColor: '#ffffff',
          borderRadius: '16px',
          padding: '1.5rem 2rem',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <div>
            <h2 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 700 }}>
              🔔 Continuous Monitoring & Alert Channels
            </h2>
            <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.8rem' }}>
              Connect Slack webhooks or Email to receive automated alerts when drift or degradation occurs.
            </p>
          </div>
          <button type="button" onClick={onClose} className="ghost-text-btn" style={{ fontSize: '1.1rem' }}>
            ✕
          </button>
        </div>

        {/* Existing Rules List */}
        <div style={{ marginBottom: '1.25rem' }}>
          <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.88rem' }}>Configured Channels</h4>
          {rules.map((r) => (
            <div
              key={r.rule_id}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '0.5rem 0.75rem',
                backgroundColor: '#f8fafc',
                borderRadius: '6px',
                marginBottom: '0.4rem',
                border: '1px solid #e2e8f0',
              }}
            >
              <div>
                <span style={{ fontWeight: 600, fontSize: '0.84rem' }}>{r.name}</span>
                <span className="muted" style={{ fontSize: '0.74rem', marginLeft: '0.5rem' }}>
                  ({r.channel_type.toUpperCase()})
                </span>
              </div>
              <button
                type="button"
                onClick={() => handleSendTest(r.rule_id)}
                className="action-btn"
                style={{ padding: '0.25rem 0.6rem', fontSize: '0.75rem' }}
              >
                ⚡ Test Webhook
              </button>
            </div>
          ))}
        </div>

        {/* New Rule Form */}
        <form onSubmit={handleSaveRule} style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div>
              <label style={{ fontSize: '0.76rem', fontWeight: 600 }}>Channel Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="horizon-input"
                style={{ width: '100%', padding: '0.35rem' }}
                required
              />
            </div>
            <div>
              <label style={{ fontSize: '0.76rem', fontWeight: 600 }}>Channel Type</label>
              <select
                value={channelType}
                onChange={(e) => setChannelType(e.target.value)}
                className="horizon-input"
                style={{ width: '100%', padding: '0.35rem' }}
              >
                <option value="slack">Slack Incoming Webhook</option>
                <option value="teams">Microsoft Teams</option>
                <option value="email">Email SMTP Dispatch</option>
              </select>
            </div>
          </div>

          <div>
            <label style={{ fontSize: '0.76rem', fontWeight: 600 }}>
              {channelType === 'email' ? 'Recipient Email Address' : 'Webhook URL'}
            </label>
            <input
              type="text"
              value={channelType === 'email' ? emailRecipient : webhookUrl}
              onChange={(e) => channelType === 'email' ? setEmailRecipient(e.target.value) : setWebhookUrl(e.target.value)}
              className="horizon-input"
              style={{ width: '100%', padding: '0.35rem' }}
              required
            />
          </div>

          <div style={{ display: 'flex', gap: '1.25rem', marginTop: '0.2rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8rem', cursor: 'pointer' }}>
              <input type="checkbox" checked={triggerDrift} onChange={(e) => setTriggerDrift(e.target.checked)} />
              Alert on High Data Drift (p &lt; 0.01)
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8rem', cursor: 'pointer' }}>
              <input type="checkbox" checked={triggerDegradation} onChange={(e) => setTriggerDegradation(e.target.checked)} />
              Alert on Performance Degradation (&gt;10%)
            </label>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', marginTop: '0.75rem' }}>
            <button type="button" onClick={onClose} className="action-btn" style={{ padding: '0.35rem 0.85rem' }}>
              Cancel
            </button>
            <button type="submit" className="primary-btn" disabled={loading} style={{ padding: '0.35rem 1rem' }}>
              Save Alert Channel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
