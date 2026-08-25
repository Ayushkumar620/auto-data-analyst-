import React from 'react';
import { PageContainer, PageHeader } from '../components/layout/PageContainer';
import { IconActivity } from '../components/ui/Icons';

type ComingSoonPageProps = {
  title: string;
  phase?: string;
  description?: string;
};

export default function ComingSoonPage({
  title,
  phase,
  description,
}: ComingSoonPageProps) {
  return (
    <PageContainer>
      <PageHeader title={title} />
      <div className="coming-soon-container">
        <div className="coming-soon-icon" aria-hidden="true">
          <IconActivity size={48} />
        </div>
        <h2 className="coming-soon-title">Module in Development</h2>
        {phase && <p className="coming-soon-phase">Planned for {phase}</p>}
        <p className="coming-soon-desc">
          {description ??
            `The ${title} module is being built and will be available in a future phase.`}
        </p>
      </div>
    </PageContainer>
  );
}
